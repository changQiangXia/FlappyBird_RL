"""
模型推理与可视化脚本
支持加载训练好的模型进行游戏演示
"""
import argparse
import os
import sys
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import make_env
from agents import DQNAgent, PPOAgent
from configs.dqn_config import get_dqn_config
from configs.ppo_config import get_ppo_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Flappy Bird RL Evaluation')
    parser.add_argument('--algo', type=str, required=True, choices=['dqn', 'ppo'],
                        help='算法: dqn 或 ppo')
    parser.add_argument('--mode', type=str, default='pixels', choices=['pixels', 'features'],
                        help='输入模式: pixels 或 features')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--episodes', type=int, default=5,
                        help='运行 episode 数量')
    parser.add_argument('--fps', type=int, default=30,
                        help='渲染帧率 (0 表示最快速度)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    
    return parser.parse_args()


def evaluate_dqn(args):
    """DQN 评估"""
    print("=" * 60)
    print("DQN 模型评估")
    print("=" * 60)
    
    use_pixels = (args.mode == 'pixels')
    config = get_dqn_config(use_pixels=use_pixels)
    config.device = args.device
    
    # 创建环境 (必须开启渲染)
    env = make_env(render_mode="human", use_pixels=use_pixels, seed=args.seed)
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # 创建智能体并加载模型
    agent = DQNAgent(obs_shape, n_actions, config, device=args.device)
    
    print(f"加载模型: {args.checkpoint}")
    agent.load(args.checkpoint)
    agent.policy_net.eval()
    
    print(f"运行 {args.episodes} 个 episode...")
    
    total_rewards = []
    total_scores = []
    
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode}/{args.episodes} 开始...")
        
        while not done:
            # 选择动作 (epsilon=0, 纯贪心)
            with torch.no_grad():
                action = agent.select_action(obs, epsilon=0.0)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # 控制帧率
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
        
        score = info.get('score', 0)
        total_rewards.append(episode_reward)
        total_scores.append(score)
        
        print(f"Episode {episode} 结束 | 奖励: {episode_reward:.2f} | 分数: {score} | 步数: {episode_length}")
    
    env.close()
    
    # 打印统计
    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    print(f"平均奖励: {sum(total_rewards)/len(total_rewards):.2f} (± {torch.tensor(total_rewards).std():.2f})")
    print(f"最高奖励: {max(total_rewards):.2f}")
    print(f"最低奖励: {min(total_rewards):.2f}")
    print(f"平均分数: {sum(total_scores)/len(total_scores):.2f}")
    print(f"最高分数: {max(total_scores)}")


def evaluate_ppo(args):
    """PPO 评估"""
    print("=" * 60)
    print("PPO 模型评估")
    print("=" * 60)
    
    use_pixels = (args.mode == 'pixels')
    config = get_ppo_config(use_pixels=use_pixels)
    config.device = args.device
    
    # 创建环境
    env = make_env(render_mode="human", use_pixels=use_pixels, seed=args.seed)
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # 创建智能体并加载模型
    agent = PPOAgent(obs_shape, n_actions, config, device=args.device)
    
    print(f"加载模型: {args.checkpoint}")
    agent.load(args.checkpoint)
    agent.network.eval()
    
    print(f"运行 {args.episodes} 个 episode...")
    
    total_rewards = []
    total_scores = []
    
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode}/{args.episodes} 开始...")
        
        while not done:
            # 选择动作
            with torch.no_grad():
                actions, _, _ = agent.select_action(obs[None, ...])
                action = actions[0]
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # 控制帧率
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
        
        score = info.get('score', 0)
        total_rewards.append(episode_reward)
        total_scores.append(score)
        
        print(f"Episode {episode} 结束 | 奖励: {episode_reward:.2f} | 分数: {score} | 步数: {episode_length}")
    
    env.close()
    
    # 打印统计
    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    print(f"平均奖励: {sum(total_rewards)/len(total_rewards):.2f} (± {torch.tensor(total_rewards).std():.2f})")
    print(f"最高奖励: {max(total_rewards):.2f}")
    print(f"最低奖励: {min(total_rewards):.2f}")
    print(f"平均分数: {sum(total_scores)/len(total_scores):.2f}")
    print(f"最高分数: {max(total_scores)}")


def main():
    """主函数"""
    args = parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    
    # 检查模型文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件不存在 {args.checkpoint}")
        return
    
    # 开始评估
    if args.algo == 'dqn':
        evaluate_dqn(args)
    else:
        evaluate_ppo(args)


if __name__ == "__main__":
    main()
