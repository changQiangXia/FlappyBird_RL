# Flappy Bird RL - Deep Reinforcement Learning Framework

**[English](#english-version) | [中文](#中文版本)**

Click above to switch language / 点击上方切换语言

---

# 中文版本

<div id="中文版本"></div>

[跳转到英文版本](#english-version)

深度强化学习训练框架，使用 DQN 和 PPO 算法训练 Flappy Bird 智能体。

<p align="center">
  <img src="https://raw.githubusercontent.com/changQiangXia/FlappyBird_RL/main/docs/training_demo.png" alt="训练界面" width="500"/>
  <br>
  <em>图1：训练时的 Pygame 可视化界面，显示分数、奖励、步数和动作信息</em>
</p>

---

## 📋 目录

1. [训练成果展示](#训练成果展示)
2. [算法实现思路](#算法实现思路)
3. [快速开始](#快速开始)
4. [详细训练指南](#详细训练指南)
5. [Checkpoint管理](#checkpoint管理)
6. [模型测试与演示](#模型测试与演示)
7. [训练技巧与调参](#训练技巧与调参)
8. [故障排除](#故障排除)

---

## 🏆 训练成果展示

### PPO 算法 100万步训练结果

<p align="center">
  <img src="https://raw.githubusercontent.com/changQiangXia/FlappyBird_RL/main/docs/ppo_1m_result.png" alt="PPO训练结果" width="600"/>
  <br>
  <em>图2：PPO算法训练100万步后的结果，最高分数达到240+，展现了优秀的持续飞行能力</em>
</p>

**关键数据：**
- 最高分数：**240+** 根管子
- 稳定分数：50-100 根管子
- 训练时间：约2-3小时
- 算法：PPO + Features模式

---

## 🧠 算法实现思路

### DQN (Deep Q-Network)

**核心架构：**
输入 -> CNN特征提取 -> Dueling结构 -> Q值输出

**关键技术点：**

| 技术 | 说明 | 作用 |
|-----|------|------|
| Double DQN | 用Policy网络选动作，Target网络算Q值 | 解决Q值过估计问题 |
| Dueling DQN | 分离Value流和Advantage流 | 更稳定地学习哪些状态有价值 |
| 经验回放 | 存储转移样本，随机采样训练 | 打破样本相关性，提高数据效率 |
| 目标网络 | 定期复制Policy网络参数 | 稳定学习目标，避免震荡 |
| Epsilon-Greedy | 以ε概率随机探索 | 平衡探索与利用 |
| 混合精度 | FP16训练，FP32梯度更新 | 加速训练，节省显存 |

**网络结构（Pixels模式）：**
```
Conv2d(4->32, 8x8, stride=4) -> ReLU
Conv2d(32->64, 4x4, stride=2) -> ReLU  
Conv2d(64->64, 3x3, stride=1) -> ReLU
Flatten -> Linear(3136->512) -> ReLU
Value流: Linear(512->1)
Advantage流: Linear(512->n_actions)
Q = Value + (Advantage - mean(Advantage))
```

**超参数：**
- 学习率：1e-4
- 折扣因子gamma：0.99
- Epsilon衰减：1.0 -> 0.05（25,000步）
- 目标网络更新频率：每1000步
- Batch size：64
- Buffer大小：50,000

---

### PPO (Proximal Policy Optimization)

**核心架构：**
输入 -> 共享CNN -> Actor(策略) + Critic(价值)

**关键技术点：**

| 技术 | 说明 | 作用 |
|-----|------|------|
| Actor-Critic | Actor输出动作概率，Critic评估状态价值 | 结合策略梯度和值函数近似 |
| GAE | 广义优势估计 | 平衡偏差与方差，稳定优势计算 |
| PPO-Clip | 限制策略更新幅度 | 防止策略突变，训练更稳定 |
| 多环境并行 | 同时运行4-8个环境 | 样本收集更快，数据更多样 |
| 正交初始化 | 网络权重正交初始化 | 改善初始梯度流 |
| 熵正则 | 最大化策略熵 | 鼓励探索，避免过早收敛 |

**网络结构：**
```
共享特征提取器（同DQN CNN）
|-- Actor: Linear(512->2) -> Softmax -> 动作概率
|-- Critic: Linear(512->1) -> 状态价值
```

**PPO-Clip目标函数：**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
其中 r(θ) = π_θ(a|s) / π_θ_old(a|s)，ε=0.2
```

**超参数：**
- 学习率：2.5e-4
- 折扣因子gamma：0.99
- GAE lambda：0.95
- 并行环境数：4（pixels）/ 8（features）
- 每环境步数：512（pixels）/ 1024（features）
- 训练轮数：4轮
- Clip范围：0.2
- 熵系数：0.01

---

## 快速开始

### 环境安装

```bash
# 创建conda环境
conda create -n flappy_rl python=3.10
conda activate flappy_rl

# 安装依赖
pip install torch torchvision gymnasium flappy-bird-gymnasium opencv-python tqdm tensorboard
```

### 3分钟快速训练

```bash
# 最快的训练方式（features模式）
python train.py --algo ppo --mode features --timesteps 50000 --render-every 10
```

预期效果：5-10分钟后，智能体能通过5-10根管子。

---

## 详细训练指南

### 训练命令结构

```bash
python train.py \
    --algo {dqn|ppo} \              # 选择算法
    --mode {pixels|features} \      # 选择输入模式
    --timesteps N \                 # 总训练步数
    --render-every M \              # 每M局演示一次
    --frame-skip K \                # 每K帧决策一次
    --seed S \                      # 随机种子
    --device {cuda|cpu}             # 训练设备
```

### 训练模式对比

| 模式 | 输入维度 | 速度 | 收敛难度 | 推荐场景 |
|-----|---------|------|---------|---------|
| **Features** | 8维向量 | ~500 it/s | 容易 | 快速验证、生产训练 |
| **Pixels** | 84x84x4图像 | ~100 it/s | 较难 | 研究、端到端学习 |

**Features模式输入：**
- 小鸟Y坐标
- 小鸟垂直速度
- 下根管道X距离
- 下根上管道Y坐标
- 下根下管道Y坐标
- 再下根管道X距离
- 再下根上管道Y坐标
- 再下根下管道Y坐标

### 推荐训练方案

**方案1：快速验证（5分钟）**
```bash
python train.py --algo ppo --mode features --timesteps 50000 --render-every 5
```
预期分数：5-10分

**方案2：稳定训练（30分钟）**
```bash
python train.py --algo ppo --mode features --timesteps 200000 --render-every 20
```
预期分数：30-50分

**方案3：冲击高分（2-3小时）**
```bash
python train.py --algo ppo --mode features --timesteps 1000000 --render-every 100
```
预期分数：100+分（实际可达240+）

**方案4：像素模式训练（较慢）**
```bash
# DQN Pixels
python train.py --algo dqn --mode pixels --timesteps 200000 --render-every 20 --frame-skip 1

# PPO Pixels
python train.py --algo ppo --mode pixels --timesteps 200000 --render-every 20
```

### 从Checkpoint恢复训练

```bash
# 继续训练
python train.py --algo ppo --mode features --timesteps 500000 \
    --resume checkpoints/PPO_features/model_latest.pt
```

---

## Checkpoint管理

### 保存位置

Checkpoint自动保存在 `./checkpoints/` 目录下：

```
checkpoints/
├── DQN_pixels/
│   ├── model_best.pt        # 最佳模型（按分数）
│   ├── model_latest.pt      # 最新模型
│   ├── model_ep100.pt       # 第100局模型
│   └── ...
├── DQN_features/
│   └── ...
├── PPO_pixels/
│   └── ...
└── PPO_features/
    ├── model_latest.pt
    ├── model_ep20.pt
    └── ...
```

### 不同算法的保存策略

| 算法 | 保存内容 | 命名规则 |
|-----|---------|---------|
| **DQN** | policy_net, target_net, optimizer, epsilon | model_best.pt, model_ep{N}.pt |
| **PPO** | network (Actor+Critic), optimizer | model_latest.pt, model_ep{N}.pt |

**注意：**
- DQN有model_best.pt（保存历史最高分模型）
- PPO只有model_latest.pt和按episode保存的模型

---

## 模型测试与演示

### 播放训练好的模型

```bash
# 基本用法
python play.py --checkpoint <路径> --algo <算法> --mode <模式>

# 播放PPO features最新模型
python play.py \
    --checkpoint checkpoints/PPO_features/model_latest.pt \
    --algo ppo \
    --mode features
```

### 常用播放命令

**播放PPO features最新模型，5局：**
```bash
python play.py --checkpoint checkpoints/PPO_features/model_latest.pt --algo ppo --mode features
```

**播放DQN features最佳模型，10局，慢速：**
```bash
python play.py --checkpoint checkpoints/DQN_features/model_best.pt --algo dqn --mode features --episodes 10 --fps 30
```

**播放DQN pixels模型：**
```bash
python play.py --checkpoint checkpoints/DQN_pixels/model_best.pt --algo dqn --mode pixels
```

### 演示时的可视化信息

播放时会显示：
- **SCORE**: 当前分数
- **REWARD**: 本局累计奖励
- **STEPS**: 已飞行步数
- **ACTION**: 当前动作（FLAP/NONE）
- **DIST**: 到管道中心的距离

---

## 训练技巧与调参

### 奖励函数设计（关键！）

当前奖励构成（envs/wrappers.py）：

```
+25.0   # 通过一根管子（核心奖励）
+2.0    # 位置奖励（高斯分布，越接近管道中心越高）
+0.5    # 在管道间隙范围内额外奖励
+0.05   # 每帧生存奖励
+0.1    # 进步奖励（管道在靠近）
-0.2    # 太靠近天花板/地板惩罚
-5.0    # 死亡惩罚
```

**调参建议：**
- 如果智能体"苟活"不过管：提高PASS_REWARD（如+50）
- 如果智能体撞天花板/地板：提高位置奖励，加强高度惩罚
- 如果智能体通过但分数不高：添加速度奖励，鼓励快速通过

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 浏览器打开 http://localhost:6006
```

**关键指标：**
- episode/score: 每局分数（最重要）
- episode/reward: 每局奖励
- step/loss: 训练损失
- demo/score: 演示分数

---

## 故障排除

### 问题1：CUDA OOM（显存不足）

**解决方案：**
```bash
python train.py --algo dqn --mode features --timesteps 100000 --device cpu
```

### 问题2：ModuleNotFoundError

```bash
conda activate flappy_rl
pip install gymnasium flappy-bird-gymnasium
```

### 问题3：分数卡在1-2分不提升

**解决方案：**
1. 尝试PPO算法（通常比DQN更易收敛）
2. 使用features模式
3. 延长训练时间（至少5万步）

---

**祝训练愉快！期待你的智能体突破100分！** 🚀🐦

---

<div id="english-version"></div>

[Back to Chinese Version](#中文版本)

---

# English Version

Deep Reinforcement Learning training framework using DQN and PPO algorithms to train Flappy Bird agents.

<p align="center">
  <img src="https://raw.githubusercontent.com/changQiangXia/FlappyBird_RL/main/docs/training_demo.png" alt="Training Interface" width="500"/>
  <br>
  <em>Figure 1: Pygame visualization interface during training, showing score, reward, steps and action information</em>
</p>

---

## Table of Contents

1. [Training Results](#training-results)
2. [Algorithm Implementation](#algorithm-implementation)
3. [Quick Start](#quick-start)
4. [Detailed Training Guide](#detailed-training-guide)
5. [Checkpoint Management](#checkpoint-management)
6. [Model Testing & Demo](#model-testing--demo)
7. [Training Tips](#training-tips)
8. [Troubleshooting](#troubleshooting)

---

## Training Results

### PPO Algorithm - 1 Million Steps

<p align="center">
  <img src="https://raw.githubusercontent.com/changQiangXia/FlappyBird_RL/main/docs/ppo_1m_result.png" alt="PPO Training Result" width="600"/>
  <br>
  <em>Figure 2: PPO algorithm results after 1 million training steps, achieving highest score of 240+ pipes, demonstrating excellent sustained flight capability</em>
</p>

**Key Metrics:**
- Highest Score: **240+** pipes
- Stable Score: 50-100 pipes
- Training Time: ~2-3 hours
- Algorithm: PPO + Features mode

---

## Algorithm Implementation

### DQN (Deep Q-Network)

**Core Architecture:**
Input -> CNN Feature Extraction -> Dueling Structure -> Q-value Output

**Key Techniques:**

| Technique | Description | Purpose |
|-----------|-------------|---------|
| Double DQN | Use Policy network to select actions, Target network to compute Q-values | Solve Q-value overestimation |
| Dueling DQN | Separate Value stream and Advantage stream | More stable state value learning |
| Experience Replay | Store transitions, sample randomly for training | Break sample correlation, improve data efficiency |
| Target Network | Periodically copy Policy network parameters | Stabilize learning target |
| Epsilon-Greedy | Random exploration with probability ε | Balance exploration and exploitation |
| Mixed Precision | FP16 training, FP32 gradient update | Speed up training, save memory |

**Network Structure (Pixels Mode):**
```
Conv2d(4->32, 8x8, stride=4) -> ReLU
Conv2d(32->64, 4x4, stride=2) -> ReLU  
Conv2d(64->64, 3x3, stride=1) -> ReLU
Flatten -> Linear(3136->512) -> ReLU
Value stream: Linear(512->1)
Advantage stream: Linear(512->n_actions)
Q = Value + (Advantage - mean(Advantage))
```

**Hyperparameters:**
- Learning rate: 1e-4
- Discount factor gamma: 0.99
- Epsilon decay: 1.0 -> 0.05 (25,000 steps)
- Target network update: every 1000 steps
- Batch size: 64
- Buffer size: 50,000

---

### PPO (Proximal Policy Optimization)

**Core Architecture:**
Input -> Shared CNN -> Actor (Policy) + Critic (Value)

**Key Techniques:**

| Technique | Description | Purpose |
|-----------|-------------|---------|
| Actor-Critic | Actor outputs action probabilities, Critic evaluates state value | Combine policy gradient and value function approximation |
| GAE | Generalized Advantage Estimation | Balance bias and variance |
| PPO-Clip | Limit policy update magnitude | Prevent policy collapse |
| Parallel Environments | Run 4-8 environments simultaneously | Faster sample collection |
| Orthogonal Initialization | Orthogonal weight initialization | Improve initial gradient flow |
| Entropy Regularization | Maximize policy entropy | Encourage exploration |

**Network Structure:**
```
Shared Feature Extractor (same as DQN CNN)
|-- Actor: Linear(512->2) -> Softmax -> Action probabilities
|-- Critic: Linear(512->1) -> State value
```

**PPO-Clip Objective:**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
where r(θ) = π_θ(a|s) / π_θ_old(a|s), ε=0.2
```

**Hyperparameters:**
- Learning rate: 2.5e-4
- Discount factor gamma: 0.99
- GAE lambda: 0.95
- Parallel environments: 4 (pixels) / 8 (features)
- Steps per environment: 512 (pixels) / 1024 (features)
- Training epochs: 4
- Clip range: 0.2
- Entropy coefficient: 0.01

---

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n flappy_rl python=3.10
conda activate flappy_rl

# Install dependencies
pip install torch torchvision gymnasium flappy-bird-gymnasium opencv-python tqdm tensorboard
```

### 3-Minute Quick Training

```bash
# Fastest training method (features mode)
python train.py --algo ppo --mode features --timesteps 50000 --render-every 10
```

Expected results: After 5-10 minutes, the agent can pass 5-10 pipes.

---

## Detailed Training Guide

### Training Command Structure

```bash
python train.py \
    --algo {dqn|ppo} \              # Select algorithm
    --mode {pixels|features} \      # Select input mode
    --timesteps N \                 # Total training steps
    --render-every M \              # Demo every M episodes
    --frame-skip K \                # Decision every K frames
    --seed S \                      # Random seed
    --device {cuda|cpu}             # Training device
```

### Training Mode Comparison

| Mode | Input Dimension | Speed | Difficulty | Recommended Scene |
|------|----------------|-------|------------|-------------------|
| **Features** | 8-dim vector | ~500 it/s | Easy | Quick validation |
| **Pixels** | 84x84x4 image | ~100 it/s | Hard | End-to-end learning |

**Features Mode Input:**
- Bird Y position
- Bird vertical velocity
- Distance to next pipe (X)
- Next upper pipe Y position
- Next lower pipe Y position
- Distance to following pipe (X)
- Following upper pipe Y position
- Following lower pipe Y position

### Recommended Training Schemes

**Scheme 1: Quick Validation (5 minutes)**
```bash
python train.py --algo ppo --mode features --timesteps 50000 --render-every 5
```
Expected score: 5-10 pipes

**Scheme 2: Stable Training (30 minutes)**
```bash
python train.py --algo ppo --mode features --timesteps 200000 --render-every 20
```
Expected score: 30-50 pipes

**Scheme 3: High Score Challenge (2-3 hours)**
```bash
python train.py --algo ppo --mode features --timesteps 1000000 --render-every 100
```
Expected score: 100+ pipes (can reach 240+)

**Scheme 4: Pixels Mode Training (slower)**
```bash
# DQN Pixels
python train.py --algo dqn --mode pixels --timesteps 200000 --render-every 20 --frame-skip 1

# PPO Pixels
python train.py --algo ppo --mode pixels --timesteps 200000 --render-every 20
```

### Resume from Checkpoint

```bash
# Continue training
python train.py --algo ppo --mode features --timesteps 500000 \
    --resume checkpoints/PPO_features/model_latest.pt
```

---

## Checkpoint Management

### Save Location

Checkpoints are automatically saved in `./checkpoints/` directory:

```
checkpoints/
├── DQN_pixels/
│   ├── model_best.pt        # Best model (by score)
│   ├── model_latest.pt      # Latest model
│   └── ...
├── PPO_features/
│   ├── model_latest.pt
│   ├── model_ep20.pt
│   └── ...
```

### Save Strategy

| Algorithm | Saved Content | Naming |
|-----------|---------------|--------|
| **DQN** | policy_net, target_net, optimizer, epsilon | model_best.pt, model_ep{N}.pt |
| **PPO** | network (Actor+Critic), optimizer | model_latest.pt, model_ep{N}.pt |

**Note:**
- DQN has model_best.pt (saves highest score model)
- PPO only has model_latest.pt and episode-based models

---

## Model Testing & Demo

### Play Trained Model

```bash
# Basic usage
python play.py --checkpoint <path> --algo <algorithm> --mode <mode>

# Play PPO features latest model
python play.py \
    --checkpoint checkpoints/PPO_features/model_latest.pt \
    --algo ppo \
    --mode features
```

### Common Play Commands

**Play PPO features latest model, 5 episodes:**
```bash
python play.py --checkpoint checkpoints/PPO_features/model_latest.pt --algo ppo --mode features
```

**Play DQN features best model, 10 episodes, slow speed:**
```bash
python play.py --checkpoint checkpoints/DQN_features/model_best.pt --algo dqn --mode features --episodes 10 --fps 30
```

### Visualization Information

During playback, the following is displayed:
- **SCORE**: Current score
- **REWARD**: Episode cumulative reward
- **STEPS**: Flight steps
- **ACTION**: Current action (FLAP/NONE)
- **DIST**: Distance to pipe center

---

## Training Tips

### Reward Function Design (Key!)

Current reward composition (envs/wrappers.py):

```
+25.0   # Pass one pipe (core reward)
+2.0    # Position reward (Gaussian, higher near pipe center)
+0.5    # Extra reward within pipe gap range
+0.05   # Per-frame survival reward
+0.1    # Progress reward (pipe approaching)
-0.2    # Too close to ceiling/floor penalty
-5.0    # Death penalty
```

**Tuning Suggestions:**
- If agent "survives" but doesn't pass pipes: Increase PASS_REWARD (e.g., +50)
- If agent hits ceiling/floor: Increase position reward, strengthen height penalty
- If agent passes but score is low: Add speed reward to encourage quick passing

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser http://localhost:6006
```

**Key Metrics:**
- episode/score: Score per episode (most important)
- episode/reward: Reward per episode
- step/loss: Training loss
- demo/score: Demo score

---

## Troubleshooting

### Issue 1: CUDA OOM

**Solution:**
```bash
python train.py --algo dqn --mode features --timesteps 100000 --device cpu
```

### Issue 2: ModuleNotFoundError

```bash
conda activate flappy_rl
pip install gymnasium flappy-bird-gymnasium
```

### Issue 3: Score Stuck at 1-2

**Solutions:**
1. Try PPO algorithm (usually converges easier than DQN)
2. Use features mode
3. Extend training time (at least 50k steps)

---

**Happy Training! Looking forward to your agent breaking 100 points!** 🚀🐦

---

## 图片说明 / Image Notes

**中文：** 请将截图保存到以下位置：
- `docs/training_demo.png` - Pygame训练界面截图
- `docs/ppo_1m_result.png` - PPO 100万步训练结果截图

**English:** Please save screenshots to:
- `docs/training_demo.png` - Pygame training interface screenshot
- `docs/ppo_1m_result.png` - PPO 1M steps training result screenshot
