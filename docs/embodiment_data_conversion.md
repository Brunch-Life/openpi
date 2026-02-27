# data.pkl -> LeRobot -> RLinf SFT（简版）

## 1) 把 `data.pkl` 转成 LeRobot
在 `openpi` 仓库执行：

```bash
cd /home/i-chenyn/data/openpi

HF_LEROBOT_HOME=/home/i-chenyn/data/RLinf/datasets/gello_teleop \
uv run scripts/convert_embodiment_pkl_to_lerobot.py \
  --input_path /home/i-chenyn/data/openpi/datasets \
  --repo_id physical-intelligence/custom_dataset \
  --task "teleoperate the robot to complete the manipulation task"
```

说明：
- 输出会写到  
  `/home/i-chenyn/data/RLinf/datasets/gello_teleop/physical-intelligence/custom_dataset`
- 当前脚本会把原始 19 维 state 转成 7 维：`pos(3)+rpy(3)+gripper(1)`。

## 2) 计算 norm stats
```bash
cd /home/i-chenyn/data/openpi

HF_LEROBOT_HOME=/home/i-chenyn/data/RLinf/datasets/gello_teleop \
uv run scripts/compute_norm_stats.py \
  --config-name pi0_custom \
  --repo-id physical-intelligence/custom_dataset
```

生成：
- `/home/i-chenyn/data/openpi/assets/pi0_custom/physical-intelligence/custom_dataset/norm_stats.json`

## 3) 把 norm stats 放到 RLinf 的模型目录
```bash
mkdir -p /home/i-chenyn/data/RLinf/checkpoints/torch/pi0_base/physical-intelligence/custom_dataset
cp /home/i-chenyn/data/openpi/assets/pi0_custom/physical-intelligence/custom_dataset/norm_stats.json \
   /home/i-chenyn/data/RLinf/checkpoints/torch/pi0_base/physical-intelligence/custom_dataset/norm_stats.json
```

## 4) 配置 RLinf 并启动 SFT
确认 `custom_sft_openpi.yaml` 里：
- `actor.model.model_path: /home/i-chenyn/data/RLinf/checkpoints/torch/pi0_base`
- `actor.model.openpi.config_name: pi0_custom`
- `data.data_path: /home/i-chenyn/data/RLinf/datasets/gello_teleop`（或你的实际路径）

启动训练：

```bash
cd /home/i-chenyn/data/RLinf
bash examples/sft/run_embodiment_sft.sh custom_sft_openpi
```
