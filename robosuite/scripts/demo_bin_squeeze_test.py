import random
import time
import cv2

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


def test_video(env, video_path='demo/test/test.mp4'):

    import imageio
    writer = imageio.get_writer(video_path, fps=20)

    episodes = 1
    # action = env.action_space.sample()
    # action[0:3] = np.array([1., 0., 0.])
    # action = np.array([0, 0, 0., 1., 1., 1., 1.])
    # left = np.array([1., 0., 0., 0., 0., 0., 0.])
    # front = np.array([0., 1., 0., 0., 0., 0., 0.])
    up = np.array([0., 0., 1.])
    down = np.array([0., 0., -1.])
    for _ in range(episodes):
        env.reset()

        arr_imgs = []
        succ = False

        for i in range(1000):
            # run a uniformly random agent
            action = env.action_space.sample()
            action[2] = -0.3
            # action = up.copy()
            # action[0:3] = 0.

            obs, reward, done, info = env.step(action)

            # contains depth
            if obs.shape[-1] == 4:
                image = obs[:, :, :-1]
                depth = obs[:, :, -1]

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                obs = np.concatenate((image, depth), 0)

            # writer.append_data(obs)
            arr_imgs.append(obs)

            if done:
                succ = (reward >= 10)
                break

        if succ:
            text = 'Success'
            color = (0, 255, 0)
        else:
            text = 'Fail'
            color = (255, 0, 0)

        for img in arr_imgs:
            cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

            writer.append_data(img)

    writer.close()


def adjust_camera_pos(env):
    env.reset()

    pos = np.array([1, 1, 1])
    for _ in range(1):
        action = env.action_space.sample()

        obs, rew, done, info = env.step(action)

        imgs = Image.fromarray(obs)
        imgs.show()

        # import ipdb
        # ipdb.set_trace()


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

    from robosuite.scripts.hard_case import get_hard_cases

    case_train, case_test = get_hard_cases()

    env = suite.make(
        'BinSqueezeMulti',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_height=128,
        camera_width=128,
        camera_depth=True,
        place_num=5,
        # random_quat=True,
        fix_rotation=True,
        total_steps=1000,
        test_cases=[],
    )


    # run(env)
    test_video(env)
    # adjust_camera_pos(env)