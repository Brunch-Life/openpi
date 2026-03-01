#!/usr/bin/env python3
"""OpenPI + SERL Franka real-robot deployment script.

Conventions in this script:
  state  (policy input): absolute [x, y, z, rx, ry, rz, gripper]
  action (policy output): relative [dx, dy, dz, drx, dry, drz, gripper_cmd]
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi.policies import policy_config
from openpi.shared import normalize as normalize_utils
from openpi.training import config as train_config


AUTO_NORM_STATS_RELATIVE_DIR = Path("physical-intelligence/custom_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy OpenPI on Franka with RealSense D435i input.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory containing model.safetensors")
    parser.add_argument("--config-name", type=str, default="pi0_custom", help="OpenPI config name")
    parser.add_argument("--robot-ip", type=str, required=True, help="Franka robot IP")
    parser.add_argument("--camera-serial", type=str, required=True, help="RealSense D435i serial number")
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

        camera = Camera(CameraInfo(name="wrist_1", serial_number=args.camera_serial))
        camera.open()
        print("Camera opened. Starting control loop. Press Ctrl+C to stop.")

        while not stop_requested:
            frame_bgr = camera.get_frame(timeout=5)
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
            _cleanup(camera, controller)
            print("Shutdown complete.")
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)


if __name__ == "__main__":
    main()
