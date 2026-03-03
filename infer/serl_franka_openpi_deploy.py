#!/usr/bin/env python3
"""OpenPI + SERL Franka real-robot deployment script.

Conventions in this script:
  state  (policy input): absolute [x, y, z, rx, ry, rz, gripper]
  action (policy output): relative [dx, dy, dz, drx, dry, drz, gripper_cmd]
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi.policies import policy_config
from openpi.shared import normalize as normalize_utils
from openpi.training import config as train_config


AUTO_NORM_STATS_RELATIVE_DIR = Path("assets/YinuoTHU/franka_real_gello")
DEFAULT_INITIAL_JOINT_POSITION = [
    0.0,
    -0.78539816339,
    0.0,
    -2.35619449019,
    0.0,
    1.57079632679,
    0.78539816339,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy OpenPI on Franka with RealSense D435i input.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory containing model.safetensors")
    parser.add_argument("--config-name", type=str, default="pi0_custom", help="OpenPI config name")
    parser.add_argument("--robot-ip", type=str, required=True, help="Franka robot IP")
    parser.add_argument("--camera-serial", type=str, default="141722078696", help="RealSense D435i serial number")
    parser.add_argument(
        "--camera-init-retries",
        type=int,
        default=15,
        help="Number of attempts to initialize the camera before aborting",
    )
    parser.add_argument(
        "--camera-init-retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between camera initialization attempts",
    )
    parser.add_argument(
        "--camera-frame-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a frame before triggering camera recovery",
    )
    parser.add_argument("--prompt", type=str, default="perform the manipulation task", help="Language prompt")
    parser.add_argument("--open-loop-steps", type=int, default=4, help="Execute first N actions per replan")
    parser.add_argument("--control-hz", type=float, default=10.0, help="Action execution frequency")
    parser.add_argument("--pos-scale", type=float, default=1.0, help="Scale for dx,dy,dz")
    parser.add_argument("--rot-scale", type=float, default=1.0, help="Scale for drx,dry,drz")
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Threshold for gripper command: >th open, < -th close",
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=Path,
        default=None,
        help=(
            "Optional norm stats directory containing norm_stats.json. "
            "If omitted, will auto-try <checkpoint-dir>/physical-intelligence/custom_dataset"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device for inference (cuda/cpu/cuda:0)")
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default="none",
        help=(
            "Torch compile mode for PI0 sampling. "
            "Use none/off to disable compile; valid compile modes include "
            "default/reduce-overhead/max-autotune."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions but do not move robot/gripper")
    parser.add_argument(
        "--initial-joint-position",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_JOINT_POSITION,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help=(
            "Initial Franka 7-DoF joint position used at startup and when pressing 'a' "
            "(immediate key press in interactive TTY)."
        ),
    )
    return parser.parse_args()


def _import_realworld_interfaces():
    from realworld.common.camera import Camera, CameraInfo
    from realworld.franka.franka_controller import FrankaController

    return Camera, CameraInfo, FrankaController


def _wait_one(ref):
    return ref.wait()[0]


def _load_policy(
    checkpoint_dir: Path,
    config_name: str,
    norm_stats_dir: Path | None,
    device: str,
):
    safetensors_path = checkpoint_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Expected safetensors checkpoint at: {safetensors_path}")

    config = train_config.get_config(config_name)

    selected_norm_stats_dir = norm_stats_dir
    if selected_norm_stats_dir is None:
        auto_dir = checkpoint_dir / AUTO_NORM_STATS_RELATIVE_DIR
        if (auto_dir / "norm_stats.json").exists():
            selected_norm_stats_dir = auto_dir
            print(f"Auto-detected norm stats: {auto_dir / 'norm_stats.json'}")

    norm_stats = normalize_utils.load(selected_norm_stats_dir) if selected_norm_stats_dir is not None else None

    return policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device=device,
    )


def _to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_bgr[..., ::-1])


def _build_absolute_state(controller) -> np.ndarray:
    robot_state = _wait_one(controller.get_state())
    tcp_pose = np.asarray(robot_state.tcp_pose, dtype=np.float32)  # xyz + quat
    rpy = R.from_quat(tcp_pose[3:].copy()).as_euler("xyz").astype(np.float32)
    gripper = np.array([float(robot_state.gripper_position)], dtype=np.float32)
    return np.concatenate([tcp_pose[:3], rpy, gripper], axis=0).astype(np.float32)


def _execute_relative_action(
    controller,
    action: np.ndarray,
    *,
    pos_scale: float,
    rot_scale: float,
    gripper_threshold: float,
    dry_run: bool,
) -> None:
    action = np.asarray(action, dtype=np.float32)
    assert action.shape == (7,), f"Expected 7D action, got {action.shape}"

    robot_state = _wait_one(controller.get_state())
    tcp_pose = np.asarray(robot_state.tcp_pose, dtype=np.float32)  # xyz + quat

    target_xyz = tcp_pose[:3] + action[:3] * pos_scale
    target_rot = R.from_euler("xyz", action[3:6] * rot_scale) * R.from_quat(tcp_pose[3:].copy())
    target_quat = target_rot.as_quat().astype(np.float32)
    target_pose = np.concatenate([target_xyz, target_quat], axis=0).astype(np.float32)

    if dry_run:
        print(f"[DRY RUN] action={action} target_pose={target_pose}")
        return

    _wait_one(controller.move_arm(target_pose))

    gripper_cmd = float(action[6])
    if gripper_cmd >= gripper_threshold:
        _wait_one(controller.open_gripper())
    elif gripper_cmd <= -gripper_threshold:
        _wait_one(controller.close_gripper())


def _cleanup(camera, controller) -> None:
    if camera is not None:
        try:
            camera.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: camera close failed: {exc}")

    if controller is not None:
        try:
            if hasattr(controller, "shutdown"):
                _wait_one(controller.shutdown())
            elif hasattr(controller, "stop_impedance"):
                _wait_one(controller.stop_impedance())
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: controller shutdown failed: {exc}")


def _open_camera_with_retry(
    Camera,
    CameraInfo,
    *,
    serial_number: str,
    retries: int,
    retry_delay: float,
):
    attempts = max(1, retries)
    delay_s = max(0.0, retry_delay)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            camera = Camera(CameraInfo(name="wrist_1", serial_number=serial_number))
            camera.open()
            if attempt > 1:
                print(f"Camera opened on retry {attempt}/{attempts}.")
            return camera
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"Camera init failed ({attempt}/{attempts}): {exc}")
            if hasattr(Camera, "list_connected_devices"):
                try:
                    devices = Camera.list_connected_devices()
                    if devices:
                        formatted_devices = ", ".join(
                            f"{d.get('serial_number', 'unknown')} ({d.get('name', 'unknown')})" for d in devices
                        )
                    else:
                        formatted_devices = "none"
                    print(f"Detected RealSense devices: {formatted_devices}")
                except Exception as list_exc:  # noqa: BLE001
                    print(f"Warning: failed to enumerate RealSense devices: {list_exc}")

            if attempt < attempts:
                time.sleep(delay_s)

    raise RuntimeError(
        f"Unable to initialize RealSense camera serial={serial_number} after {attempts} attempt(s)."
    ) from last_exc


def _reset_to_initial_joint(
    controller,
    joint_position: Sequence[float],
    *,
    dry_run: bool,
) -> None:
    target_joint = np.asarray(joint_position, dtype=np.float32).reshape(-1)
    if target_joint.shape != (7,):
        raise ValueError(f"Expected 7D initial joint position, got {target_joint.shape}")

    if dry_run:
        print(f"[DRY RUN] reset_joint -> {target_joint.tolist()}")
        return

    _wait_one(controller.reset_joint(target_joint.tolist()))


def _keyboard_command_worker(
    *,
    stop_event: threading.Event,
    execution_event: threading.Event,
    reset_event: threading.Event,
    quit_event: threading.Event,
) -> None:
    if not sys.stdin.isatty():
        return

    stdin_fd = sys.stdin.fileno()
    original_termios = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)
    try:
        while not stop_event.is_set() and not quit_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue

            raw = os.read(stdin_fd, 1)
            if not raw:
                continue

            cmd = raw.decode(errors="ignore").lower()
            if cmd == "a":
                reset_event.set()
                execution_event.clear()
                print("\n[KEY] a -> reset to initial position", flush=True)
            elif cmd == "c":
                if not execution_event.is_set():
                    print("\n[KEY] c -> start/continue execution", flush=True)
                execution_event.set()
            elif cmd == "q":
                quit_event.set()
                print("\n[KEY] q -> quit", flush=True)
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_termios)


def main() -> None:
    args = parse_args()
    os.environ["OPENPI_TORCH_COMPILE_MODE"] = args.torch_compile_mode

    Camera, CameraInfo, FrankaController = _import_realworld_interfaces()
    policy = _load_policy(args.checkpoint_dir, args.config_name, args.norm_stats_dir, args.device)

    print("Launching Franka controller...")
    controller = FrankaController.launch_controller(robot_ip=args.robot_ip)
    camera = None
    stop_requested = False
    signal_count = 0
    interactive_control = False
    keyboard_thread: threading.Thread | None = None
    keyboard_stop_event: threading.Event | None = None
    execution_event: threading.Event | None = None
    reset_event: threading.Event | None = None
    quit_event: threading.Event | None = None

    def _request_stop(signum, _frame):  # noqa: ANN001
        nonlocal signal_count, stop_requested
        signal_count += 1
        stop_requested = True
        signal_name = signal.Signals(signum).name
        if signal_count == 1:
            print(f"Received {signal_name}. Stopping...")
            raise KeyboardInterrupt
        else:
            print(f"Received {signal_name} again. Forcing exit.")
            raise SystemExit(130)

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        print("Waiting for robot to become ready...")
        while not stop_requested and not bool(_wait_one(controller.is_robot_up())):
            time.sleep(0.5)

        if stop_requested:
            return

        camera = _open_camera_with_retry(
            Camera,
            CameraInfo,
            serial_number=args.camera_serial,
            retries=args.camera_init_retries,
            retry_delay=args.camera_init_retry_delay,
        )
        print("Camera opened.")
        print("Moving robot to initial joint position...")
        _reset_to_initial_joint(
            controller,
            args.initial_joint_position,
            dry_run=args.dry_run,
        )

        interactive_control = sys.stdin.isatty()
        if interactive_control:
            execution_event = threading.Event()
            reset_event = threading.Event()
            quit_event = threading.Event()
            keyboard_stop_event = threading.Event()
            keyboard_thread = threading.Thread(
                target=_keyboard_command_worker,
                kwargs={
                    "stop_event": keyboard_stop_event,
                    "execution_event": execution_event,
                    "reset_event": reset_event,
                    "quit_event": quit_event,
                },
                daemon=True,
                name="keyboard-command-worker",
            )
            keyboard_thread.start()

            print("Interactive commands enabled (no Enter needed):")
            print("  - Press 'a' to reset to initial joint position")
            print("  - Press 'c' to start/continue policy execution")
            print("  - Press 'q' to quit")
            print("Waiting for 'c' to start execution...")
        else:
            print("stdin is not a TTY. Starting control loop immediately.")

        while not stop_requested:
            if interactive_control:
                assert execution_event is not None
                assert reset_event is not None
                assert quit_event is not None

                if quit_event.is_set():
                    print("Quit command received.")
                    break

                if reset_event.is_set():
                    reset_event.clear()
                    print("Reset command received. Moving to initial joint position...")
                    _reset_to_initial_joint(
                        controller,
                        args.initial_joint_position,
                        dry_run=args.dry_run,
                    )
                    print("Robot reset complete. Press 'c' to start execution.")
                    continue

                if not execution_event.is_set():
                    time.sleep(0.05)
                    continue

            try:
                frame_bgr = camera.get_frame(timeout=args.camera_frame_timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to read camera frame: {exc}")
                try:
                    camera.close()
                except Exception as close_exc:  # noqa: BLE001
                    print(f"Warning: camera close during recovery failed: {close_exc}")
                camera = _open_camera_with_retry(
                    Camera,
                    CameraInfo,
                    serial_number=args.camera_serial,
                    retries=args.camera_init_retries,
                    retry_delay=args.camera_init_retry_delay,
                )
                print("Camera recovered. Waiting for next control cycle...")
                time.sleep(0.05)
                continue
            image_rgb = _to_rgb(frame_bgr)
            state_abs = _build_absolute_state(controller)

            obs = {
                "observation/image": image_rgb,
                "observation/state": state_abs,
                "prompt": args.prompt,
            }
            action_chunk = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
            planned_actions = action_chunk[: args.open_loop_steps]

            print(f"infer -> action_chunk.shape={action_chunk.shape}, execute={planned_actions.shape[0]}")

            for action in planned_actions:
                if stop_requested:
                    break
                if interactive_control:
                    assert reset_event is not None
                    assert quit_event is not None

                    if quit_event.is_set():
                        print("Quit command received.")
                        stop_requested = True
                        break
                    if reset_event.is_set():
                        reset_event.clear()
                        print("Reset command received during execution.")
                        _reset_to_initial_joint(
                            controller,
                            args.initial_joint_position,
                            dry_run=args.dry_run,
                        )
                        print("Robot reset complete. Press 'c' to continue.")
                        break

                step_start = time.time()
                _execute_relative_action(
                    controller,
                    action,
                    pos_scale=args.pos_scale,
                    rot_scale=args.rot_scale,
                    gripper_threshold=args.gripper_threshold,
                    dry_run=args.dry_run,
                )
                elapsed = time.time() - step_start
                time.sleep(max(0.0, (1.0 / args.control_hz) - elapsed))
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            if keyboard_stop_event is not None:
                keyboard_stop_event.set()
            if keyboard_thread is not None:
                keyboard_thread.join(timeout=1.0)
            _cleanup(camera, controller)
            print("Shutdown complete.")
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)


if __name__ == "__main__":
    main()
