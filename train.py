"""
训练入口脚本 - 重构优化版
支持 DQN 和 PPO，支持像素/特征输入，支持自动演示
"""
import argparse
import os
import sys
import torch
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import make_env
from agents import DQNAgent, PPOAgent
from configs.dqn_config import get_dqn_config
from configs.ppo_config import get_ppo_config
from utils import Logger, save_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Flappy Bird RL Training')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='算法选择: dqn 或 ppo')
    parser.add_argument('--mode', type=str, default='pixels', choices=['pixels', 'features'],
                        help='输入模式: pixels (84x84x4) 或 features (8维特征)')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='总训练步数')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Frame skip 步数（每n帧决策一次，默认1）')
    parser.add_argument('--render-every', type=int, default=20,
                        help='每N个episode自动演示一次（0表示不演示，默认20）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='TensorBoard 日志目录')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备: cuda 或 cpu')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练（路径）')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def evaluate_and_render(agent, env_config, device, n_episodes=1, fps=60, algo='dqn'):
    """评估智能体并渲染演示"""
    render_env = make_env(
        render_mode="human",
        use_pixels=env_config['use_pixels'],
        frame_skip=1,
        use_annotated_render=True,
        seed=env_config['seed'] + 1000
    )
    
    total_rewards = []
    total_scores = []
    
    print(f"\n{'='*60}")
    print(f"🎮 演示模式（{n_episodes} 个 episode）")
    print(f"{'='*60}")
    
    for ep in range(n_episodes):
        obs, info = render_env.reset(seed=env_config['seed'] + 1000 + ep)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if algo == 'dqn':
                action = agent.select_action(obs, epsilon=0.0)
            else:
                # PPO
                with torch.no_grad():
                    actions, _, _ = agent.select_action(obs[None, ...])
                    action = actions[0]
            
            obs, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            time.sleep(1.0 / fps)
        
        score = info.get('score', 0)
        total_rewards.append(episode_reward)
        total_scores.append(score)
        
        print(f"  Demo Ep {ep+1}: Reward={episode_reward:+.2f} | Score={score} | Steps={episode_length}")
    
    render_env.close()
    
    avg_reward = np.mean(total_rewards)
    avg_score = np.mean(total_scores)
    
    print(f"  平均: Reward={avg_reward:+.2f} | Score={avg_score:.1f}")
    print(f"{'='*60}\n")
    
    return avg_reward, avg_score


def train_dqn(args):
    """DQN 训练流程 - 优化版"""
    print("=" * 60)
    print("🚀 DQN 训练开始")
    print(f"   输入模式: {args.mode}")
    print(f"   Frame Skip: {args.frame_skip}")
    print(f"   演示间隔: 每 {args.render_every} episode")
    print(f"   设备: {args.device}")
    print("=" * 60)
    
    use_pixels = (args.mode == 'pixels')
    config = get_dqn_config(use_pixels=use_pixels)
    config.total_timesteps = args.timesteps
    config.device = args.device
    
    # 创建训练环境
    env = make_env(
        render_mode=None,
        use_pixels=use_pixels,
        frame_skip=args.frame_skip,
        use_annotated_render=False,
        seed=args.seed
    )
    
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"观察空间: {obs_shape}, 动作空间: {n_actions}")
    
    # 创建智能体
    agent = DQNAgent(obs_shape, n_actions, config, device=args.device)
    print(f"网络参数: {sum(p.numel() for p in agent.parameters()):,}")
    
    # 加载checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n📂 加载 checkpoint: {args.resume}")
        agent.load(args.resume, reset_epsilon=0.3)
    
    # 创建日志器
    logger = Logger(args.log_dir, f"DQN_{args.mode}")
    
    # 训练循环
    obs, info = env.reset(seed=args.seed)
    episode = 0
    episode_reward = 0
    episode_length = 0
    best_reward = -float('inf')
    best_score = 0
    total_steps = 0
    
    demo_env_config = {'use_pixels': use_pixels, 'seed': args.seed}
    
    pbar = tqdm(total=args.timesteps, desc="Training DQN", ncols=100)
    
    while total_steps < args.timesteps:
        # 选择动作
        action = agent.select_action(obs, epsilon=agent.epsilon)
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存储转移
        current_score = info.get('score', 0)
        agent.store_transition(obs, action, reward, next_obs, done, score=current_score)
        
        episode_reward += reward
        episode_length += 1
        total_steps += 1
        obs = next_obs
        
        # 学习
        if total_steps >= config.learning_starts and total_steps % config.train_freq == 0:
            loss_dict = agent.learn(config.batch_size if use_pixels else config.batch_size_feature)
            if total_steps % 1000 == 0:
                logger.log_step(total_steps, loss_dict)
        
        # 更新目标网络
        if total_steps % config.target_update_freq == 0 and total_steps > config.learning_starts:
            agent.update_target_network()
        
        # Episode 结束
        if done:
            episode += 1
            score = info.get('score', 0)
            
            # 记录指标
            logger.log_episode(episode, {
                'reward': episode_reward,
                'length': episode_length,
                'score': score,
                'epsilon': agent.epsilon
            })
            
            # 更新最佳
            if episode_reward > best_reward:
                best_reward = episode_reward
            if score > best_score:
                best_score = score
                save_checkpoint(
                    agent.policy_net.state_dict(),
                    agent.optimizer.state_dict(),
                    episode, total_steps,
                    os.path.join(args.save_dir, f"DQN_{args.mode}"),
                    is_best=True
                )
            
            # 定期保存
            if episode % config.save_interval == 0:
                save_checkpoint(
                    agent.policy_net.state_dict(),
                    agent.optimizer.state_dict(),
                    episode, total_steps,
                    os.path.join(args.save_dir, f"DQN_{args.mode}")
                )
            
            # 打印进度
            if episode % config.log_interval == 0:
                tqdm.write(f"Ep {episode:5d} | Step {total_steps:6d} | "
                          f"Reward: {episode_reward:+7.2f} | Score: {score:2d} | "
                          f"Best: {best_score:2d} | ε: {agent.epsilon:.3f}")
            
            # 自动演示
            if args.render_every > 0 and episode % args.render_every == 0:
                demo_reward, demo_score = evaluate_and_render(
                    agent, demo_env_config, args.device, n_episodes=1, fps=60, algo='dqn'
                )
                logger.log_scalar('demo/reward', demo_reward, episode)
                logger.log_scalar('demo/score', demo_score, episode)
            
            # 重置
            obs, info = env.reset(seed=args.seed + episode)
            episode_reward = 0
            episode_length = 0
        
        pbar.update(1)
    
    pbar.close()
    env.close()
    logger.close()
    
    print(f"\n{'='*60}")
    print(f"✅ 训练完成！")
    print(f"   总 episode: {episode}")
    print(f"   总步数: {total_steps}")
    print(f"   最佳奖励: {best_reward:+.2f}")
    print(f"   最高分数: {best_score}")
    print(f"{'='*60}")


def train_ppo(args):
    """PPO 训练流程 - 优化版"""
    print("=" * 60)
    print("🚀 PPO 训练开始")
    print(f"   输入模式: {args.mode}")
    print(f"   Frame Skip: {args.frame_skip}")
    print(f"   演示间隔: 每 {args.render_every} episode")
    print(f"   设备: {args.device}")
    print("=" * 60)
    
    use_pixels = (args.mode == 'pixels')
    config = get_ppo_config(use_pixels=use_pixels)
    config.total_timesteps = args.timesteps
    config.device = args.device
    
    n_envs = config.n_envs
    
    # 创建多个环境
    envs = [
        make_env(
            render_mode=None,
            use_pixels=use_pixels,
            frame_skip=args.frame_skip,
            use_annotated_render=False,
            seed=args.seed + i
        ) for i in range(n_envs)
    ]
    
    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n
    print(f"观察空间: {obs_shape}, 动作空间: {n_actions}")
    print(f"并行环境数: {n_envs}")
    
    # 创建智能体
    agent = PPOAgent(obs_shape, n_actions, config, device=args.device)
    print(f"网络参数: {sum(p.numel() for p in agent.parameters()):,}")
    
    # 加载checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n📂 加载 checkpoint: {args.resume}")
        agent.load(args.resume)
    
    # 创建日志器
    logger = Logger(args.log_dir, f"PPO_{args.mode}")
    
    # 初始化环境
    obs_list = [env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]
    obs_batch = np.array(obs_list)
    
    episode_rewards = [0.0] * n_envs
    episode_lengths = [0] * n_envs
    total_episodes = 0
    best_reward = -float('inf')
    best_score = 0
    total_steps = 0
    
    demo_env_config = {'use_pixels': use_pixels, 'seed': args.seed}
    
    n_steps = config.n_steps if use_pixels else config.n_steps_feature
    total_updates = args.timesteps // (n_steps * n_envs)
    
    pbar = tqdm(total=total_updates, desc="Training PPO", ncols=100)
    
    for update in range(1, total_updates + 1):
        # 收集经验
        for step in range(n_steps):
            actions, log_probs, values = agent.select_action(obs_batch)
            
            next_obs_list = []
            rewards = np.zeros(n_envs, dtype=np.float32)
            dones = np.zeros(n_envs, dtype=np.float32)
            
            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated
                
                next_obs_list.append(next_obs)
                rewards[i] = reward
                dones[i] = float(done)
                
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                if done:
                    total_episodes += 1
                    score = info.get('score', 0)
                    
                    logger.log_episode(total_episodes, {
                        'reward': episode_rewards[i],
                        'length': episode_lengths[i],
                        'score': score
                    })
                    
                    if episode_rewards[i] > best_reward:
                        best_reward = episode_rewards[i]
                    if score > best_score:
                        best_score = score
                    
                    if total_episodes % config.log_interval == 0:
                        tqdm.write(f"Ep {total_episodes:5d} | Update {update:4d} | "
                                  f"Reward: {episode_rewards[i]:+7.2f} | Score: {score:2d} | "
                                  f"Best: {best_score:2d}")
                    
                    # 自动演示
                    if args.render_every > 0 and total_episodes % args.render_every == 0:
                        demo_reward, demo_score = evaluate_and_render(
                            agent, demo_env_config, args.device, n_episodes=1, fps=60, algo='ppo'
                        )
                        logger.log_scalar('demo/reward', demo_reward, total_episodes)
                        logger.log_scalar('demo/score', demo_score, total_episodes)
                    
                    obs_reset, _ = env.reset(seed=args.seed + total_episodes + i)
                    next_obs_list[i] = obs_reset
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
            
            agent.store_transition(obs_batch, actions, log_probs, rewards, values, dones)
            obs_batch = np.array(next_obs_list)
            total_steps += n_envs
        
        # 计算优势并更新
        agent.compute_advantages(obs_batch)
        loss_dict = agent.learn()
        logger.log_step(update, loss_dict)
        
        # 定期保存
        if update % config.save_interval == 0:
            save_checkpoint(
                agent.network.state_dict(),
                agent.optimizer.state_dict(),
                update, total_steps,
                os.path.join(args.save_dir, f"PPO_{args.mode}")
            )
        
        pbar.update(1)
    
    pbar.close()
    
    for env in envs:
        env.close()
    logger.close()
    
    print(f"\n{'='*60}")
    print(f"✅ 训练完成！")
    print(f"   总更新: {total_updates}")
    print(f"   总 episode: {total_episodes}")
    print(f"   最佳奖励: {best_reward:+.2f}")
    print(f"   最高分数: {best_score}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，切换到 CPU")
        args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.algo == 'dqn':
        train_dqn(args)
    else:
        train_ppo(args)


if __name__ == "__main__":
    main()
