import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from collections import deque

"""
BOILER PLATE FILE
"""


# Set cv2 resizing interpolation method
cv2.setNumThreads(0)

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert the image observation from RGB to gray scale.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Add a channel dimension
        return obs[:, :, np.newaxis]


class ResizeObservation(gym.ObservationWrapper):
    """
    Resize the image observation.
    """
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = tuple(shape)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.shape + (1,),  # Grayscale
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = cv2.resize(
            obs, self.shape, interpolation=cv2.INTER_AREA
        )
        # Add a channel dimension
        return obs[:, :, np.newaxis]


class FrameStack(gym.ObservationWrapper):
    """
    Stack k last frames.
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], k), # k channels
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Concatenate frames along the channel axis
        return np.concatenate(self.frames, axis=2)
