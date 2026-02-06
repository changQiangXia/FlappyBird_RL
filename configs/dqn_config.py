"""
DQN 超参数配置 - 针对 RTX 3050ti 4GB 显存优化
"""
from dataclasses import dataclass


@dataclass
class DQNConfig:
    """DQN 配置类 - 优化版 V2"""
    
    # 环境参数
    use_pixels: bool = True
    
    # 网络结构
    hidden_dim: int = 512
    
    # 训练参数
    total_timesteps: int = 500000
    learning_rate: float = 1e-4  # 更稳定的学习率
    gamma: float = 0.99
    
    # epsilon-greedy 探索 - 快速衰减，更快进入利用
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 25000  # 2.5万步降到5%，加速收敛
    
    # Replay Buffer
    buffer_size: int = 50000
    buffer_size_feature: int = 100000
    batch_size: int = 64
    batch_size_feature: int = 128
    
    # 目标网络更新
    target_update_freq: int = 1000
    
    # 学习相关
    learning_starts: int = 500  # 500步后开始学习，更快开始
    train_freq: int = 4
    
    # 混合精度训练
    use_amp: bool = True
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 50
    
    # 设备
    device: str = "cuda"
    
    def __post_init__(self):
        if not self.use_pixels:
            self.buffer_size = self.buffer_size_feature
            self.batch_size = self.batch_size_feature


def get_dqn_config(use_pixels: bool = True) -> DQNConfig:
    return DQNConfig(use_pixels=use_pixels)
