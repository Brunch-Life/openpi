"""
Convert RLinf embodiment `data.pkl` files to LeRobot format for OpenPI training.

Example:
uv run scripts/convert_embodiment_pkl_to_lerobot.py \
    --input_path datasets \
    --repo_id your_hf_username/embodiment_franka \
    --task "pick up the object and place it into the container"
"""

from pathlib import Path
import pickle
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tyro


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_image_uint8(image) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape={image.shape}")

    # Convert CHW -> HWC if needed.
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if np.issubdtype(image.dtype, np.floating):
        min_v = float(np.nanmin(image))
        max_v = float(np.nanmax(image))
        if min_v >= -1e-6 and max_v <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    else:
        image = np.clip(image, 0, 255)

    image = image.astype(np.uint8)

    # Convert grayscale to RGB.
    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)

    if image.shape[-1] != 3:
        raise ValueError(f"Expected image with 3 channels, got shape={image.shape}")

    return image


def _to_state_pos_rpy_gripper7(state) -> np.ndarray:
    """Convert state to 7D [pos(3), rpy(3), gripper(1)] for Franka OpenPI training.

    Supported input formats:
    - 19D state from RLinf realworld collector after wrappers:
      [gripper(1), force(3), pose_xyz_rpy(6), torque(3), vel(6)]
    - 7D state already in [pos, rpy, gripper]
    """
    state = np.asarray(_to_numpy(state), dtype=np.float32).reshape(-1)
    if state.shape == (19,):
        # 19D layout in RealWorldEnv (sorted by key):
        # [gripper(1), tcp_force(3), tcp_pose_xyz_rpy(6), tcp_torque(3), tcp_vel(6)]
        pos_rpy = state[4:10]
        gripper = state[0:1]
        return np.concatenate([pos_rpy, gripper], axis=0)
    if state.shape == (7,):
        return state
    raise ValueError(
        f"Unsupported state shape {state.shape}. Expected (19,) or (7,)."
    )


def _is_episode_end(step: dict) -> bool:
    for key in ("dones", "terminations", "truncations"):
        if key not in step:
            continue
        value = _to_numpy(step[key]).reshape(-1)
        if value.size > 0 and bool(value[0]):
            return True
    return False


def _find_pkl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("data.pkl"))


def _load_steps(pkl_path: Path) -> list[dict]:
    with pkl_path.open("rb") as f:
        steps = pickle.load(f)
    if not isinstance(steps, list):
        raise ValueError(f"{pkl_path} is not a list, got type={type(steps)}")
    return steps


def main(
    input_path: str = "datasets",
    *,
    repo_id: str = "your_hf_username/embodiment_franka",
    task: str = "perform the manipulation task",
    fps: int = 10,
    robot_type: str = "franka",
    overwrite: bool = True,
    push_to_hub: bool = False,
):
    input_path = Path(input_path)
    pkl_files = _find_pkl_files(input_path)
    if not pkl_files:
        raise FileNotFoundError(f"No data.pkl found under: {input_path}")

    first_nonempty_steps = None
    for pkl_file in pkl_files:
        steps = _load_steps(pkl_file)
        if steps:
            first_nonempty_steps = steps
            break
    if first_nonempty_steps is None:
        raise ValueError("All input pkl files are empty.")

    first_step = first_nonempty_steps[0]
    obs = first_step["transitions"]["obs"]
    image_shape = _to_image_uint8(obs["main_images"]).shape
    state_shape = tuple(_to_state_pos_rpy_gripper7(obs["states"]).shape)
    action_shape = tuple(_to_numpy(first_step["action"]).shape)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"{output_path} exists. Set --overwrite to replace it.")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            # This dataset has only one camera, so we duplicate the same view as wrist image
            # to stay compatible with OpenPI Libero-style data pipeline.
            "wrist_image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    num_steps = 0
    num_episodes = 0
    for pkl_file in pkl_files:
        steps = _load_steps(pkl_file)
        if not steps:
            continue

        for step in steps:
            obs = step["transitions"]["obs"]
            image = _to_image_uint8(obs["main_images"])
            state = _to_state_pos_rpy_gripper7(obs["states"])
            action = np.asarray(_to_numpy(step["action"]), dtype=np.float32)

            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": image,
                    "state": state,
                    "actions": action,
                    "task": task,
                }
            )
            num_steps += 1

            if _is_episode_end(step):
                dataset.save_episode()
                num_episodes += 1

        # If source file ends without done/termination/truncation, force close episode.
        if not _is_episode_end(steps[-1]):
            dataset.save_episode()
            num_episodes += 1

    print(f"Converted {len(pkl_files)} pkl file(s)")
    print(f"Steps: {num_steps}")
    print(f"Episodes: {num_episodes}")
    print(f"Saved LeRobot dataset to: {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["openpi", "franka", "embodiment"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
