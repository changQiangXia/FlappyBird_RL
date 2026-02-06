"""
智能体基类
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseAgent(ABC, nn.Module):
    """强化学习智能体基类"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = False  # 自动混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
    
    @abstractmethod
    def select_action(self, observation, epsilon: float = 0.0) -> int:
        """选择动作"""
        pass
    
    @abstractmethod
    def learn(self, *args, **kwargs) -> dict:
        """学习/训练一步"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移动到设备"""
        return tensor.to(self.device, non_blocking=True)
