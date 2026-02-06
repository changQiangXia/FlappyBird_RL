"""
运行训练好的模型并渲染观看 - 简化版
"""
import argparse
import torch
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import make_env
from agents import DQNAgent, PPOAgent
from configs.dqn_config import get_dqn_config
from configs.ppo_config import get_ppo_config
from utils import load_checkpoint


def play(agent_type, checkpoint_path, n_episodes=5, fps=60, mode='pixels'):
    """播放训练好的模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_pixels = (mode == 'pixels')
    
    # 创建环境
    env = make_env(
        render_mode="human",
        use_pixels=use_pixels,
        frame_skip=1,
        use_annotated_render=True,
        seed=42
    )
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # 创建智能体
    if agent_type == 'dqn':
        config = get_dqn_config(use_pixels=use_pixels)
        agent = DQNAgent(obs_shape, n_actions, config, device=device)
        # 使用 utils.load_checkpoint 加载
        load_checkpoint(checkpoint_path, agent.policy_net, agent.optimizer)
        # 同步 target_net
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    else:
        config = get_ppo_config(use_pixels=use_pixels)
        agent = PPOAgent(obs_shape, n_actions, config, device=device)
        load_checkpoint(checkpoint_path, agent.network, agent.optimizer)
    
    print(f"\n{'='*60}")
    print(f"🎮 开始播放 {n_episodes} 个 episode")
    print(f"   模型: {checkpoint_path}")
    print(f"   按 Ctrl+C 退出")
    print(f"{'='*60}\n")
    
    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=42 + ep)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                if agent_type == 'dqn':
                    action = agent.select_action(obs, epsilon=0.0)
                else:
                    with torch.no_grad():
                        actions, _, _ = agent.select_action(obs[None, ...])
                        action = actions[0]
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                time.sleep(1.0 / fps)
            
            score = info.get('score', 0)
            print(f"Episode {ep+1}: Reward={episode_reward:+.2f} | Score={score} | Steps={episode_length}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    env.close()
    print("\n播放结束")


def main():
    parser = argparse.ArgumentParser(description='Play Flappy Bird RL')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型路径')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'])
    parser.add_argument('--mode', type=str, default='pixels', choices=['pixels', 'features'])
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--fps', type=int, default=60)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        return
    
    play(args.algo, args.checkpoint, args.episodes, args.fps, args.mode)


if __name__ == "__main__":
    main()
