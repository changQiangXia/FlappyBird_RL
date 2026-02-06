"""
PPO 超参数配置 - 针对 RTX 3050ti 4GB 显存优化
"""
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO 配置类 - 优化版"""
    
    # 环境参数
    use_pixels: bool = True
    n_envs: int = 4  # 并行环境数
    
    # 网络结构
    hidden_dim: int = 512
    
    # 训练参数
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO 特定参数
    n_steps: int = 512  # 每环境收集步数
    n_steps_feature: int = 1024
    batch_size: int = 64
    batch_size_feature: int = 128
    n_epochs: int = 4  # 每次更新训练轮数
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01  # 适度熵正则，鼓励探索
    max_grad_norm: float = 0.5
    
    # 混合精度训练
    use_amp: bool = True
    
    # 学习率衰减
    use_lr_decay: bool = True
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 20
    
    # 设备
    device: str = "cuda"
    
    def __post_init__(self):
        if not self.use_pixels:
            self.n_steps = self.n_steps_feature
            self.batch_size = self.batch_size_feature
            self.n_envs = 8


def get_ppo_config(use_pixels: bool = True) -> PPOConfig:
    return PPOConfig(use_pixels=use_pixels)
