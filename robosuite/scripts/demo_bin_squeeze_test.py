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


def test_step(env):
    obs = env.reset()

    action = env.action_space.sample()
    env.step(action)


def test_qpos_meaning(env):
    # create a video writer with imageio
    # writer = imageio.get_writer(video_path, fps=20)

    obs = env.reset()
    imgs = Image.fromarray(obs)
    imgs.show()

    action = np.array([0.2, 0.3])
    obs, rew, done, info = env.step(action)
    imgs = Image.fromarray(obs)
    imgs.show()


def test_video(env, video_path='demo/test/test_random.mp4'):
    obs = env.reset()

    import imageio
    writer = imageio.get_writer(video_path, fps=20)

    for i in range(10000):
        # run a uniformly random agent
        # action = np.array([-1, 0.1])
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        writer.append_data(obs)
        print("Saving frame #{}".format(i))

        if done:
            break

    writer.close()


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

    hard_case = {
        'obj_names': ['Can', 'Can'],
        'obj_poses': [
            np.array([-0.03, 0.03]),
            np.array([0.03, 0.03]),
            # np.array([-0.03, 0]),
            # np.array([0.03, 0]),
            # np.array([0, 0, 0.5]),
        ],
        'target_object': 'Can1'
    }

    env = suite.make(
        'BinSqueeze',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        action_pos_index=np.array([2, 5]),
        # hard_case=hard_case,
    )


    # run(env)
    test_video(env)