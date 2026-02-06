"""
模型检查点保存和加载
"""
import torch
import os
from typing import Optional, Dict, Any


def save_checkpoint(
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    episode: int,
    step: int,
    save_dir: str,
    prefix: str = "model",
    is_best: bool = False
):
    """
    保存模型检查点
    
    Args:
        model_state: 模型状态字典
        optimizer_state: 优化器状态字典
        episode: 当前 episode
        step: 当前 step
        save_dir: 保存目录
        prefix: 文件名前缀
        is_best: 是否为最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'episode': episode,
        'step': step,
    }
    
    # 保存最新模型
    latest_path = os.path.join(save_dir, f"{prefix}_latest.pt")
    torch.save(checkpoint, latest_path)
    
    # 保存指定 episode 的模型
    episode_path = os.path.join(save_dir, f"{prefix}_ep{episode}.pt")
    torch.save(checkpoint, episode_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(save_dir, f"{prefix}_best.pt")
        torch.save(checkpoint, best_path)
    
    print(f"[Checkpoint] 已保存到 {episode_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例 (可选)
    
    Returns:
        包含 episode 和 step 的字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    episode = checkpoint.get('episode', 0)
    step = checkpoint.get('step', 0)
    
    print(f"[Checkpoint] 已从 {checkpoint_path} 加载 (Episode {episode}, Step {step})")
    
    return {'episode': episode, 'step': step}


def get_latest_checkpoint(save_dir: str, prefix: str = "model") -> Optional[str]:
    """获取最新的检查点文件路径"""
    latest_path = os.path.join(save_dir, f"{prefix}_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    return None
