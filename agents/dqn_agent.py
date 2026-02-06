"""
DQN Agent - 优化版
支持: Double DQN, Dueling DQN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional
import random

from agents.base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """DQN 网络 - Dueling架构"""
    
    def __init__(self, input_shape: Tuple, n_actions: int, hidden_dim: int = 512, use_pixels: bool = True):
        super().__init__()
        self.use_pixels = use_pixels
        
        if use_pixels:
            self.features = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            feature_size = 64 * 7 * 7
        else:
            feature_size = input_shape[0]
            self.features = nn.Sequential(
                nn.Linear(feature_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            feature_size = hidden_dim
        
        # Dueling DQN
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """经验回放缓冲区 - 优化版"""
    
    def __init__(self, capacity: int, obs_shape: Tuple, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        self.is_pixel = len(obs_shape) > 1
        dtype = np.uint8 if self.is_pixel else np.float32
        
        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        """存储经验"""
        if self.is_pixel and obs.dtype == np.float32:
            obs = (obs * 255).astype(np.uint8)
            next_obs = (next_obs * 255).astype(np.uint8)
        
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        if self.is_pixel:
            obs = torch.from_numpy(self.obs[indices]).float().to(self.device) / 255.0
            next_obs = torch.from_numpy(self.next_obs[indices]).float().to(self.device) / 255.0
        else:
            obs = torch.from_numpy(self.obs[indices]).to(self.device)
            next_obs = torch.from_numpy(self.next_obs[indices]).to(self.device)
        
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return self.size


class DQNAgent(BaseAgent):
    """DQN 智能体 - 优化版"""
    
    def __init__(self, obs_shape: Tuple, n_actions: int, config, device: str = "cuda"):
        super().__init__(device)
        
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.config = config
        
        # 创建网络
        self.policy_net = DQNNetwork(obs_shape, n_actions, config.hidden_dim, config.use_pixels).to(self.device)
        self.target_net = DQNNetwork(obs_shape, n_actions, config.hidden_dim, config.use_pixels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        # 经验回放缓冲区
        buffer_size = config.buffer_size if config.use_pixels else config.buffer_size_feature
        self.replay_buffer = ReplayBuffer(buffer_size, obs_shape, device)
        
        # 混合精度
        self.use_amp = config.use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # epsilon-greedy
        self.epsilon = config.epsilon_start
        self.epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps
        self.epsilon_end = config.epsilon_end
        
        self.steps_done = 0
    
    def select_action(self, observation: np.ndarray, epsilon: Optional[float] = None) -> int:
        """使用 epsilon-greedy 选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        
        # 线性衰减 epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        if random.random() > epsilon:
            with torch.no_grad():
                # 确保输入是 float32，与 AMP 兼容
                obs_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    q_values = self.policy_net(obs_tensor)
                return q_values.argmax(dim=1).item()
        else:
            return random.randrange(self.n_actions)
    
    def learn(self, batch_size: int) -> Dict[str, float]:
        """训练一步"""
        if len(self.replay_buffer) < batch_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 当前 Q 值
            current_q = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Double DQN: 用policy_net选动作，target_net算Q值
            with torch.no_grad():
                next_actions = self.policy_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
                target_q = rewards + self.config.gamma * next_q * (1 - dones)
            
            # Huber Loss
            loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()
        
        self.steps_done += 1
        
        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'epsilon': self.epsilon
        }
    
    def update_target_network(self):
        """硬更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool, score: int = 0):
        """存储转移"""
        self.replay_buffer.push(obs, action, reward, next_obs, done)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str, reset_epsilon: Optional[float] = None):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        
        if reset_epsilon is not None:
            self.epsilon = reset_epsilon
        else:
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
