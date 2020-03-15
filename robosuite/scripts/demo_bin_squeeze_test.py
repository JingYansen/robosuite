import random
import time

import numpy as np
import robosuite as suite

from robosuite.wrappers import MyGymWrapper
from PIL import Image

def test_render_time(env):
    env.reset()

    ts = []
    ts_step = []
    for _ in range(30):

        action = env.action_space.sample()

        beg_tiem = time.time()
        env.step(action)
        end_time = time.time()

        ts_step.append(end_time - beg_tiem)

        beg_tiem = time.time()
        env.sim.render(width=256, height=256, camera_name='birdview')
        # env.sim.render(width=256, height=256, camera_name='agentview')
        end_time = time.time()

        ts.append(end_time - beg_tiem)

    ts = np.array(ts)
    ts_step = np.array(ts_step)

    print('Time: ', ts.mean())
    print('Time step: ', ts_step.mean())


def test_env_init(env):
    obs = env.reset()

    imgs = Image.fromarray(obs)
    imgs.show()


def adjust_camera_pos(env):
    env.reset()

    pos = np.array([1, 1, 1])
    for _ in range(30):
        action = env.action_space.sample()

        obs, rew, done, info = env.step(action)

        imgs = Image.fromarray(obs)
        imgs.show()

        import ipdb
        ipdb.set_trace()


def run(env):

    for _ in range(30):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)

        import ipdb
        ipdb.set_trace()


def test_random_take(env):
    for _ in range(10):
        env.reset()

        for __ in range(100):
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)

            for i in range(200):
                env.render()

            if done:
                break


if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0.57, 0.35])
    high = np.array([0.63, 0.405])
    obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2) * 2

    env = suite.make(
        'BinSqueeze',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=1,

        # obj_names=obj_names,
        action_bound=(low, high)
    )


    # run(env)
    test_env_init(env)