"""
PPO Agent - 优化版
支持: GAE, 多环境并行, 自动混合精度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict

from agents.base_agent import BaseAgent


class PPONetwork(nn.Module):
    """PPO 网络 (Actor-Critic)"""
    
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
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """PPO 经验缓冲区"""
    
    def __init__(self, n_steps: int, n_envs: int, obs_shape: Tuple, device: str = "cuda"):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.pos = 0
        
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
    
    def add(self, obs: np.ndarray, action: np.ndarray, log_prob: np.ndarray, 
            reward: np.ndarray, value: np.ndarray, done: np.ndarray):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        self.pos += 1
    
    def compute_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """计算 GAE 优势"""
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        returns = advantages + self.values
        return advantages, returns
    
    def get_batches(self, batch_size: int):
        """生成训练批次"""
        obs = torch.from_numpy(self.obs.reshape(-1, *self.obs.shape[2:])).to(self.device)
        actions = torch.from_numpy(self.actions.reshape(-1)).to(self.device)
        log_probs = torch.from_numpy(self.log_probs.reshape(-1)).to(self.device)
        advantages = torch.from_numpy(self.advantages.reshape(-1)).to(self.device)
        returns = torch.from_numpy(self.returns.reshape(-1)).to(self.device)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        indices = np.arange(len(obs))
        np.random.shuffle(indices)
        
        for start in range(0, len(obs), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield obs[batch_idx], actions[batch_idx], log_probs[batch_idx], \
                  advantages[batch_idx], returns[batch_idx]
    
    def clear(self):
        self.pos = 0


class PPOAgent(BaseAgent):
    """PPO 智能体"""
    
    def __init__(self, obs_shape: Tuple, n_actions: int, config, device: str = "cuda"):
        super().__init__(device)
        
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.config = config
        
        # 创建网络
        self.network = PPONetwork(obs_shape, n_actions, config.hidden_dim, config.use_pixels).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # 经验缓冲区
        n_steps = config.n_steps if config.use_pixels else config.n_steps_feature
        self.buffer = RolloutBuffer(n_steps, config.n_envs, obs_shape, device)
        self.buffer.advantages = None
        self.buffer.returns = None
        
        # 混合精度
        self.use_amp = config.use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.update_count = 0
    
    def select_action(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """选择动作"""
        with torch.no_grad():
            # 确保输入是 float32，与 AMP 兼容
            obs_tensor = torch.from_numpy(observation).float().to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, log_prob: np.ndarray,
                         reward: np.ndarray, value: np.ndarray, done: np.ndarray):
        self.buffer.add(obs, action, log_prob, reward, value, done)
    
    def compute_advantages(self, last_obs: np.ndarray):
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(last_obs).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                _, _, _, last_values = self.network.get_action_and_value(last_obs_tensor)
            last_values = last_values.cpu().numpy()
        
        advantages, returns = self.buffer.compute_advantages(
            last_values, self.config.gamma, self.config.gae_lambda
        )
        self.buffer.advantages = advantages
        self.buffer.returns = returns
    
    def learn(self) -> Dict[str, float]:
        """PPO 训练"""
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(
                self.config.batch_size if self.config.use_pixels else self.config.batch_size_feature
            ):
                obs, actions, old_log_probs, advantages, returns = batch
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _, log_probs, entropy, values = self.network.get_action_and_value(obs, actions)
                    
                    # PPO-Clip
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值损失
                    value_loss = F.mse_loss(values, returns)
                    
                    # 总损失
                    loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        self.buffer.clear()
        self.update_count += 1
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def save(self, path: str):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
