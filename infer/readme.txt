我需要你做一个读取pi0的sft训练好的ckpt，然后使用pytorch进行推理，得到的数据交给真机serl去训练的真机部署程序，请你按照以下步骤来做：

1. 先实现一个读取pi0ckpt的demo，实现给定图像输入得到对应的输出，ckpt路径如下/home/i-chenyn/data/RLinf/logs/20260226-09:12:22/test_openpi/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt
    具体你可以参考https://github.com/Physical-Intelligence/openpi 里的相应文档以及如下事例实现：
    from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]

2. 写一个程序实现使用serl来控制franka机械臂，并且和前面的结合起来，输入的图像是从realsense d435i读取。

请你根据以上要求完成任务，如果有不确定的地方及时问我

要求代码尽量简洁，不需要大量的鲁棒、安全检查等

