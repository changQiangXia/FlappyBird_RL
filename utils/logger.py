"""
TensorBoard 日志记录器
"""
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
import os


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, algo_name: str):
        """
        Args:
            log_dir: 日志保存目录
            algo_name: 算法名称 (DQN 或 PPO)
        """
        self.log_dir = os.path.join(log_dir, algo_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.episode = 0
        self.step = 0
        
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录标量值"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """记录多个标量值"""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """记录一个 episode 的指标"""
        self.episode = episode
        for key, value in metrics.items():
            self.log_scalar(f"episode/{key}", value, episode)
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """记录一个 step 的指标"""
        self.step = step
        for key, value in metrics.items():
            self.log_scalar(f"step/{key}", value, step)
    
    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """记录超参数和对应指标"""
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """关闭 logger"""
        self.writer.close()
