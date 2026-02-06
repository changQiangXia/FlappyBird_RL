"""
精简版包装器 - 高性能 + 精细奖励
"""
import gymnasium as gym
import numpy as np
import cv2
from collections import deque


class PixelObservationWrapper(gym.Wrapper):
    """像素观察包装器 - 优化版"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(512, 288, 3), dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        _, _, _, _, info = self.env.step(0)
        frame = self.env.render()
        return frame, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        return frame, reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """灰度化 - 使用OpenCV快速转换"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8
        )
    
    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """resize到84x84"""
    
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
    
    def observation(self, obs):
        return cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)


class FrameStack(gym.ObservationWrapper):
    """帧堆叠 - 4帧"""
    
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        return np.array(self.frames, dtype=np.uint8)


class NormalizeObservation(gym.ObservationWrapper):
    """归一化到 [0, 1]"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )
    
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class FrameSkip(gym.Wrapper):
    """跳帧 - 每n帧做一次决策"""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        max_score = 0
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            max_score = max(max_score, info.get('score', 0))
            if terminated or truncated:
                break
        
        info['score'] = max_score
        return obs, total_reward, terminated, truncated, info


class AdvancedShapedReward(gym.Wrapper):
    """
    精细化奖励函数 - V2版本 (加强版)
    
    核心改进：
    1. 大幅提高通过奖励 - 让通过管子成为最有吸引力的行为
    2. 加强位置引导 - 明确告诉智能体哪里是安全的
    3. 优化死亡惩罚 - 不过度惩罚，鼓励尝试
    4. 添加进步奖励 - 鼓励向前飞
    
    奖励构成：
    1. 通过管道奖励: +25 (主要目标，必须足够高)
    2. 生存奖励: +0.05/frame (保持基础生存动机)
    3. 位置奖励: 基于与管道间隙中心的距离 (强引导)
    4. 进步奖励: 通过管道x坐标给予正向激励
    5. 死亡惩罚: -5 (适度惩罚，不过度保守)
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0
        self.prev_y = None
        self.prev_pipe_x = None
        self.episode_steps = 0
        
        # 游戏常量
        self.SCREEN_HEIGHT = 512
        self.PLAYER_X = 57
        self.PIPE_WIDTH = 52
        
        # 奖励系数 (V2调整)
        self.PASS_REWARD = 25.0         # 大幅提高！通过一根管子给25
        self.SURVIVAL_REWARD = 0.05     # 降低生存奖励，避免苟活
        self.DEATH_PENALTY = -5.0       # 适度死亡惩罚
        
    def reset(self, **kwargs):
        self.prev_score = 0
        self.prev_y = None
        self.prev_pipe_x = None
        self.episode_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_steps += 1
        reward = 0.0
        
        # 获取原始环境状态
        raw = self._get_raw_env()
        bird_y = getattr(raw, '_player_y', None)
        upper_pipes = getattr(raw, '_upper_pipes', [])
        lower_pipes = getattr(raw, '_lower_pipes', [])
        score = info.get('score', 0)
        
        # 1. 通过管道奖励 - 核心奖励，大幅提高！
        if score > self.prev_score:
            reward += self.PASS_REWARD
            self.prev_score = score
        
        # 2. 生存奖励 - 降低，避免智能体只追求苟活
        reward += self.SURVIVAL_REWARD
        
        # 3. 位置奖励 - 强引导智能体保持在安全区域
        if not terminated and bird_y is not None and upper_pipes and lower_pipes:
            target_pipe = self._get_target_pipe(upper_pipes, lower_pipes)
            
            if target_pipe:
                gap_center = target_pipe['gap_center']
                gap_top = target_pipe['gap_top']
                gap_bottom = target_pipe['gap_bottom']
                pipe_x = target_pipe['x']
                
                # 计算到间隙中心的距离
                dist_to_center = abs(bird_y - gap_center)
                
                # 强位置奖励：越接近中心奖励越高
                # 使用高斯函数，sigma=30意味着在中心±30像素内奖励接近最大值
                sigma = 30.0
                position_reward = 2.0 * np.exp(-(dist_to_center ** 2) / (2 * sigma ** 2))
                reward += position_reward
                
                # 额外：如果在间隙范围内（上下管道之间），再给奖励
                if gap_top < bird_y < gap_bottom:
                    reward += 0.5  # 在管道间隙内额外奖励
                
                # 4. 进步奖励 - 鼓励向前飞
                if self.prev_pipe_x is not None and pipe_x < self.prev_pipe_x:
                    # 管道在靠近，给一点鼓励
                    reward += 0.1
                self.prev_pipe_x = pipe_x
                
                # 5. 高度惩罚 - 如果太靠近天花板或地板，给惩罚
                if bird_y < 50:  # 太靠近天花板
                    reward -= 0.2
                elif bird_y > self.SCREEN_HEIGHT - 100:  # 太靠近地板
                    reward -= 0.2
                
                self.prev_y = bird_y
        
        # 6. 死亡惩罚
        if terminated or truncated:
            reward = self.DEATH_PENALTY
            
            # 根据死亡位置给予不同程度的惩罚
            if bird_y is not None:
                if bird_y < 0:  # 撞天花板
                    reward -= 2.0
                elif bird_y > self.SCREEN_HEIGHT - 50:  # 撞地板
                    reward -= 2.0
        
        return obs, reward, terminated, truncated, info
    
    def _get_raw_env(self):
        """获取原始环境"""
        raw = self.env
        while hasattr(raw, 'env'):
            raw = raw.env
        return raw
    
    def _get_target_pipe(self, upper_pipes, lower_pipes):
        """获取小鸟前方的目标管道"""
        if not upper_pipes or not lower_pipes:
            return None
        
        for up, low in zip(upper_pipes, lower_pipes):
            if up['x'] + self.PIPE_WIDTH > self.PLAYER_X:
                gap_top = up['y'] + 320
                gap_bottom = low['y']
                gap_center = (gap_top + gap_bottom) / 2
                return {
                    'x': up['x'],
                    'gap_top': gap_top,
                    'gap_bottom': gap_bottom,
                    'gap_center': gap_center
                }
        return None


class AnnotatedRender(gym.Wrapper):
    """
    带详细标注的渲染器
    显示：分数、奖励、距离中心、动作等
    """
    
    # 屏幕尺寸常量
    SCREEN_WIDTH = 288
    SCREEN_HEIGHT = 512
    
    def __init__(self, env, fps=30):
        super().__init__(env)
        self.fps = fps
        self.window_name = "Flappy Bird RL Training"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.episode_reward = 0
        self.episode_length = 0
        
    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_length = 0
        obs, info = self.env.reset(**kwargs)
        self._render_frame(info, 0, True)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        self._render_frame(info, action, terminated or truncated)
        return obs, reward, terminated, truncated, info
    
    def _render_frame(self, info, action, done):
        """渲染带标注的帧"""
        raw_env = self._get_raw_env()
        frame = raw_env.render()
        
        if frame is None:
            return
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        score = info.get('score', 0)
        
        # 左侧面板 - 游戏状态
        texts = [
            f"SCORE: {score}",
            f"REWARD: {self.episode_reward:+.2f}",
            f"STEPS: {self.episode_length}",
            f"ACTION: {'FLAP' if action == 1 else 'NONE'}",
        ]
        
        # 添加管道距离信息
        raw = self.env
        while hasattr(raw, 'env') and not hasattr(raw, '_get_target_pipe'):
            raw = raw.env
        
        if hasattr(raw, '_get_target_pipe'):
            bird_y = getattr(raw_env, '_player_y', 0)
            upper_pipes = getattr(raw_env, '_upper_pipes', [])
            lower_pipes = getattr(raw_env, '_lower_pipes', [])
            target = raw._get_target_pipe(upper_pipes, lower_pipes)
            if target and bird_y:
                dist = abs(bird_y - target['gap_center'])
                texts.append(f"DIST: {dist:.0f}px")
        
        y_offset = 30
        for text in texts:
            cv2.putText(frame_bgr, text, (10, y_offset), self.font, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # 如果结束，显示原因
        if done:
            cv2.putText(frame_bgr, "DONE!", (self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT // 2), 
                       self.font, 1.0, (0, 0, 255), 3)
        
        cv2.imshow(self.window_name, frame_bgr)
        cv2.waitKey(max(1, int(1000 / self.fps)))
    
    def _get_raw_env(self):
        raw = self.env
        while hasattr(raw, 'env'):
            raw = raw.env
        return raw
    
    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass  # 窗口可能已被销毁或从未创建
        super().close()


class HumanRender(gym.Wrapper):
    """简单渲染器 - 仅显示游戏画面"""
    
    def __init__(self, env, fps=30):
        super().__init__(env)
        self.fps = fps
        self.window_name = "Flappy Bird"
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._render_frame()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._render_frame()
        return obs, reward, terminated, truncated, info
    
    def _render_frame(self):
        raw_env = self.env.unwrapped
        frame = raw_env.render()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, frame_bgr)
            cv2.waitKey(max(1, int(1000 / self.fps)))
    
    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        super().close()
