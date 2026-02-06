"""
Flappy Bird 环境封装 - 精简优化版
"""
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings

try:
    from envs.wrappers import (
        PixelObservationWrapper,
        GrayScaleObservation,
        ResizeObservation,
        FrameStack,
        NormalizeObservation,
        FrameSkip,
        AdvancedShapedReward,
        AnnotatedRender,
        HumanRender
    )
except ImportError:
    from wrappers import (
        PixelObservationWrapper,
        GrayScaleObservation,
        ResizeObservation,
        FrameStack,
        NormalizeObservation,
        FrameSkip,
        AdvancedShapedReward,
        AnnotatedRender,
        HumanRender
    )


class NormalizeFeatureObservation(gym.ObservationWrapper):
    """特征观察值归一化"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        return obs.astype(np.float32)


class FlappyBirdEnv:
    """
    Flappy Bird 环境封装
    
    包装器顺序（重要！）：
    1. AdvancedShapedReward (最内层，紧挨原始环境)
    2. FrameSkip (在奖励之后，累积塑形后的奖励)
    3. Pixel/Feature 处理
    4. Render wrappers (最外层)
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        use_pixels: bool = True,
        frame_skip: int = 1,
        use_annotated_render: bool = True,
        seed: Optional[int] = None
    ):
        self.render_mode = render_mode
        self.use_pixels = use_pixels
        self.frame_skip = frame_skip
        self.use_annotated_render = use_annotated_render
        self.seed = seed
        
        # 创建基础环境
        self.env = self._create_base_env()
        
        # 应用包装器
        self.env = self._apply_wrappers(self.env)
        
        # 获取观察空间和动作空间
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def _create_base_env(self) -> gym.Env:
        """创建基础 Flappy Bird 环境"""
        env_id = "FlappyBird-v0"
        
        try:
            render_mode = "rgb_array" if self.use_pixels else self.render_mode
            
            env = gym.make(
                env_id,
                render_mode=render_mode,
                use_lidar=False,
            )
        except Exception as e:
            warnings.warn(f"无法创建环境 {env_id}: {e}")
            raise
        
        if self.seed is not None:
            env.reset(seed=self.seed)
        
        return env
    
    def _apply_wrappers(self, env: gym.Env) -> gym.Env:
        """应用包装器 - 注意顺序！"""
        
        # 1. 最内层：精细奖励塑形
        env = AdvancedShapedReward(env)
        
        # 2. Frame Skip
        if self.frame_skip > 1:
            env = FrameSkip(env, skip=self.frame_skip)
        
        # 3. 像素/特征处理
        if self.use_pixels:
            # 像素模式：需要 render 获取图像
            env = PixelObservationWrapper(env)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, shape=(84, 84))
            env = NormalizeObservation(env)
            env = FrameStack(env, n_frames=4)
        else:
            # 特征模式：直接归一化
            env = NormalizeFeatureObservation(env)
        
        # 4. 渲染（最外层）
        if self.render_mode == "human":
            if self.use_annotated_render:
                env = AnnotatedRender(env)
            else:
                env = HumanRender(env)
        
        return env
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(**kwargs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)
    
    def close(self):
        self.env.close()
    
    def render(self):
        return self.env.render()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_env(
    render_mode: Optional[str] = None,
    use_pixels: bool = True,
    frame_skip: int = 1,
    use_annotated_render: bool = True,
    seed: Optional[int] = None
) -> FlappyBirdEnv:
    """创建 Flappy Bird 环境的工厂函数"""
    return FlappyBirdEnv(
        render_mode=render_mode,
        use_pixels=use_pixels,
        frame_skip=frame_skip,
        use_annotated_render=use_annotated_render,
        seed=seed
    )
