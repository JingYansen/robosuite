import tqdm
import numpy as np
import robosuite as suite

from robosuite.scripts.utils import make_vec_env, norm_depth


if __name__=='__main__':

    env_id, env_type = 'BinPack-v0', 'mujoco'
    env_kwargs = {
        'has_offscreen_renderer': True,
        'use_camera_obs': True,
        'control_freq': 1,
        'camera_height': 64,
        'camera_width': 64,
        'video_height': 64,
        'video_width': 64,
        'make_dataset': True,
        'dataset_path': 'data/random',
        'action_bound': (np.array([0.5, 0.3]), np.array([0.7, 0.5]))
    }

    env_nums = 32
    env = make_vec_env(env_id, env_type, env_nums, None, env_kwargs=env_kwargs)

    # env = suite.make(
    #     'BinPackPlace',
    #     has_renderer=False,
    #     has_offscreen_renderer=True,
    #     ignore_done=True,
    #     use_camera_obs=True,
    #     control_freq=1,
    #     camera_height=64,
    #     camera_width=64,
    #     video_height=64,
    #     video_width=64,
    #     make_dataset=True,
    #     dataset_path='data/random',
    #     action_bound=(np.array([0.5, 0.3]), np.array([0.7, 0.5]))
    # )

    episodes = 1200000 // (6 * env_nums)

    progress_bar = tqdm.tqdm(desc='Collect data ', total=episodes)
    for i in range(episodes):
        progress_bar.update(1)

        env.reset()
        for _ in range(6):
            actions = []
            for __ in range(env_nums):
                actions.append(env.action_space.sample())

            obs, reward, done, info = env.step(actions)