#!/usr/bin/env python3
"""Minimal OpenPI safetensors inference demo.

State convention in this script:
  state = [x, y, z, rx, ry, rz, gripper]  (absolute pose/state)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from openpi.policies import policy_config
from openpi.shared import normalize as normalize_utils
from openpi.training import config as train_config


AUTO_NORM_STATS_RELATIVE_DIR = Path("physical-intelligence/custom_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenPI inference from a safetensors checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory containing model.safetensors")
    parser.add_argument("--config-name", type=str, default="pi0_custom", help="OpenPI config name")
    parser.add_argument("--image-path", type=Path, required=True, help="Input RGB image path")
    parser.add_argument(
        "--state",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ", "GRIPPER"),
        required=True,
        help="Absolute 7D state: x y z rx ry rz gripper",
    )
    parser.add_argument("--prompt", type=str, default="perform the manipulation task", help="Language prompt")
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
    return parser.parse_args()


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def resolve_norm_stats_dir(checkpoint_dir: Path, user_norm_stats_dir: Path | None) -> Path | None:
    if user_norm_stats_dir is not None:
        return user_norm_stats_dir

    auto_dir = checkpoint_dir / AUTO_NORM_STATS_RELATIVE_DIR
    if (auto_dir / "norm_stats.json").exists():
        print(f"Auto-detected norm stats: {auto_dir / 'norm_stats.json'}")
        return auto_dir

    return None


def main() -> None:
    args = parse_args()

    safetensors_path = args.checkpoint_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Expected safetensors checkpoint at: {safetensors_path}")

    config = train_config.get_config(args.config_name)

    norm_stats_dir = resolve_norm_stats_dir(args.checkpoint_dir, args.norm_stats_dir)
    norm_stats = normalize_utils.load(norm_stats_dir) if norm_stats_dir is not None else None

    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device=args.device,
    )

    obs = {
        "observation/image": load_rgb_image(args.image_path),
        "observation/state": np.asarray(args.state, dtype=np.float32),
        "prompt": args.prompt,
    }

    outputs = policy.infer(obs)
    actions = np.asarray(outputs["actions"])

    np.set_printoptions(suppress=True, precision=5)
    print(f"actions.shape: {actions.shape}")
    print("actions[:4]:")
    print(actions[:4])
    print("first_action:")
    print(actions[0])


if __name__ == "__main__":
    main()
