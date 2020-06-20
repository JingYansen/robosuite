import tqdm
import numpy as np
import robosuite as suite
from gym import spaces

from robosuite.scripts.utils import make_vec_env, norm_depth


if __name__=='__main__':

    env_id, env_type = 'BinPack-v0', 'mujoco'
    take_nums = 8
    env_kwargs = {
        'has_offscreen_renderer': True,
        'use_camera_obs': True,
        'control_freq': 1,
        'camera_height': 64,
        'camera_width': 64,
        'video_height': 64,
        'video_width': 64,
        'make_dataset': True,
        'random_take': True,
        'dataset_path': 'data/8obj_half_in_1m',
        'take_nums': take_nums,
        'action_bound': (np.array([0.5, 0.3]), np.array([0.67, 0.45]))
    }

    env_nums = 32
    env = make_vec_env(env_id, env_type, env_nums, None, env_kwargs=env_kwargs)

    episodes = 1000000 // (take_nums * env_nums)

    progress_bar = tqdm.tqdm(desc='Collect data ', total=episodes)
    for i in range(episodes // 2):
        progress_bar.update(1)

        env.reset()
        for _ in range(6):
            actions = []
            for __ in range(env_nums):
                actions.append(env.action_space.sample())

            obs, reward, done, info = env.step(actions)

    smaller_bound = spaces.Box(low=np.array([0.5575, 0.3375]), high=np.array([0.6425, 0.4225]))
    for i in range(episodes // 2, episodes):
        progress_bar.update(1)

        env.reset()
        for _ in range(6):
            actions = []
            for __ in range(env_nums):
                actions.append(smaller_bound.sample())

            obs, reward, done, info = env.step(actions)