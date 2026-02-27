# OpenPI Infer Scripts (Absolute State + Relative Action)

This folder provides two scripts:

1. `pi0_safetensors_demo.py`
- Offline inference demo with one image and one 7D absolute state.

2. `serl_franka_openpi_deploy.py`
- Real robot loop: D435i image -> OpenPI infer -> Franka execution.

## Conventions

- `state` (policy input): absolute 7D
  - `[x, y, z, rx, ry, rz, gripper]`
- `action` (policy output): relative 7D
  - `[dx, dy, dz, drx, dry, drz, gripper_cmd]`
- Replan policy: execute first `4` actions (configurable by `--open-loop-steps`) and then infer again.

## Prerequisites

1. Checkpoint directory contains `model.safetensors`:
- `<checkpoint_dir>/model.safetensors`

2. Optional normalization stats:
- By default, scripts auto-try:
  - `<checkpoint_dir>/physical-intelligence/custom_dataset/norm_stats.json`
- If your norm stats are elsewhere, pass:
  - `--norm-stats-dir /path/to/norm_stats_parent_dir`

3. For real robot script:
- ROS + Franka controller environment ready.
- ROS Python dependencies available (`rospy`, `geometry_msgs`, `franka_msgs`, `franka_gripper`, `serl_franka_controllers`, etc.).
- RealSense D435i connected, and you know camera serial number.

## 1) Offline Demo

```bash
cd /home/i-chenyn/data/openpi

uv run infer/pi0_safetensors_demo.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --config-name pi0_custom \
  --image-path /path/to/image.png \
  --state 0.50 0.00 0.20 -3.14 0.00 0.00 0.0 \
  --prompt "perform the manipulation task" \
  --device cuda
```

If norm stats are not in the auto-detected path, specify explicitly:

```bash
uv run infer/pi0_safetensors_demo.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --config-name pi0_custom \
  --image-path /path/to/image.png \
  --state 0.50 0.00 0.20 -3.14 0.00 0.00 0.0 \
  --norm-stats-dir /path/to/assets/physical-intelligence/custom_dataset \
  --device cuda
```

## 2) Real Robot Deploy (Dry Run)

```bash
cd /home/i-chenyn/data/openpi

uv run infer/serl_franka_openpi_deploy.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --config-name pi0_custom \
  --robot-ip <FRANKA_IP> \
  --camera-serial <D435I_SERIAL> \
  --prompt "perform the manipulation task" \
  --open-loop-steps 4 \
  --control-hz 10 \
  --dry-run
```

## 3) Real Robot Deploy (Execute Actions)

```bash
cd /home/i-chenyn/data/openpi

uv run infer/serl_franka_openpi_deploy.py \
  --checkpoint-dir /path/to/checkpoint_dir \
  --config-name pi0_custom \
  --robot-ip <FRANKA_IP> \
  --camera-serial <D435I_SERIAL> \
  --prompt "perform the manipulation task" \
  --open-loop-steps 4 \
  --control-hz 10 \
  --pos-scale 1.0 \
  --rot-scale 1.0 \
  --gripper-threshold 0.5
```

Stop with `Ctrl+C`.
