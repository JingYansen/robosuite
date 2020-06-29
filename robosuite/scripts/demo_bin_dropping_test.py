import random
import time
import cv2

import numpy as np
import robosuite as suite

from robosuite.wrappers import MyGymWrapper
from robosuite.scripts.utils import norm_depth
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


def test_video_bin_pack(env, video_path='demo/test/test.mp4'):

    import imageio
    writer = imageio.get_writer(video_path, fps=20)

    episodes = 1
    avg_reward = 0
    for _ in range(episodes):
        env.reset()
        total_reward = 0

        for i in range(env.take_nums):
            # run a uniformly random agent
            action = env.action_space.sample()
            # action = np.array([0.5, 0.3])

            obs, reward, done, info = env.step(action)
            avg_reward += reward
            total_reward += reward

            for o in info['birdview']:
                # contains depth

                image, depth = o
                depth = norm_depth(depth)

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                view = np.concatenate((image, depth), 0)

                text = str(total_reward)
                cv2.putText(view, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                writer.append_data(view)

            if done:
                break

    writer.close()
    print('avg reward:', avg_reward / episodes)


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


def test_random_res(env):
    episodes = 1600

    avg_reward = 0
    for _ in range(episodes):
        env.reset()
        total_reward = 0

        for i in range(env.take_nums):
            # run a uniformly random agent
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            avg_reward += reward
            total_reward += reward

            if done:
                break

    print('avg reward:', avg_reward / episodes)


def test_mapping_from_action_to_img(env):
    for _ in range(1):
        env.reset()
        total_reward = 0

        for i in range(env.take_nums):
            # run a uniformly random agent
            # action = env.action_space.sample()
            # action = np.array([0.5575, 0.3375])
            # action = np.array([0.6425, 0.4225])
            action = np.array([0.5575, 0.3375])

            obs, reward, done, info = env.step(action)

            for o in info['birdview']:
                # contains depth

                image, depth = o
                depth = norm_depth(depth)

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                view = np.concatenate((image, depth), 0)

                point_size = 1
                point_color = (0, 0, 255) # BGR
                thickness = 4
                # point = (11, 16)
                point = (11, 46) # [0.5575, 0.3375]
                # point = (41, 16) # [0.6425, 0.4225]
                # point = (0, 0)
                cv2.circle(view, point, point_size, point_color, thickness)
                Image.fromarray(view).save('temp.jpeg')

                return


if __name__ == "__main__":

    from robosuite.scripts.hard_case import get_hard_cases

    case_train, case_test = get_hard_cases()

    env = suite.make(
        'BinPackPlace',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=1,
        render_drop_freq=20,
        camera_height=128,
        camera_width=128,
        video_height=128,
        video_width=128,
        random_take=True
        )


    # run(env)
    test_video_bin_pack(env)
    # test_mapping_from_action_to_img(env)
    # test_random_res(env)