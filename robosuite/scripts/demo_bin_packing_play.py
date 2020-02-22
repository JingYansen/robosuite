import os
import tensorflow as tf
import numpy as np
from gym.wrappers import TimeLimit
from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.bench import Monitor
from baselines.common.vec_env.vec_monitor import VecEnvWrapper
from robosuite.environments.bin_pack_place import BinPackPlace
from gym import spaces

import random
import robosuite as suite
from robosuite.wrappers import MyGymWrapper
from PIL import Image

if __name__ == "__main__":

    low = np.array([0.57, 0.35])
    high = np.array([0.63, 0.405])
    obj_names = ['Milk'] * 1 + ['Bread'] * 1 + ['Cereal'] * 2 + ['Can'] * 2

    env = suite.make(
        'BinPackPlace',
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=1,
        obj_names=obj_names
    )

    env.viewer.set_camera(camera_id=0)


    n_episode = 10

    for i in range(n_episode):
        env.reset()
        for j in range(100):
            for _ in range(200):
                env.render()

            action = env.action_space.sample()
            print('action: ', action)
            obs, rew, done, _ = env.step(action)

            if done:
                for _ in range(200):
                    env.render()

                break
