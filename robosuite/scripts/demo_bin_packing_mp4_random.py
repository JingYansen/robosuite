import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper
from baselines.ppo2 import ppo2

import os
import numpy as np
from PIL import Image
import subprocess

DEMO_PATH = 'demo'

if not os.path.exists(DEMO_PATH):
    os.makedirs(DEMO_PATH)

DEMO_PATH += '/random_banana_bowl.mp4'

class human_policy:
    def __init__(self, low, high, delta=[0.061, 0.072]):
        self.low = np.copy(low)
        self.high = np.copy(high)
        self.delta = delta
        self.index = 0
        self.actions = []

        self.prepare()

    def prepare(self):
        for i in range(3):
            for j in range(4):
                action = self.low + self.delta * np.array([i, j])

                self.actions.append(action)

        for i in range(2):
            for j in range(2):
                action = self.low + (self.delta + np.array([0.03, 0.035])) * np.array([i, j])

                self.actions.append(action)



    def step(self):
        action = self.actions[self.index]
        self.index += 1
        return np.array(action)


def make_video(env, n_episode):
    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    subprocess.call(['mkdir', '-p', 'demo'])
    time_step_counter = 0

    ## make video
    for i_episode in range(n_episode):
        env.reset()

        for i in range(100):

            actions = env.action_space.sample()

            obs, rew, done, info = env.step(actions)

            for i in range(len(info['birdview'])):
                image_data_bird, image_data_agent = info['birdview'][i], info['targetview'][i]
                image_data = np.concatenate((image_data_bird, image_data_agent), 1)

                img = Image.fromarray(image_data, 'RGB')
                img.save('frames/frame-%.10d.png' % time_step_counter)
                time_step_counter += 1

            if done:
                break

    subprocess.call(
        ['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '24', '-pix_fmt', 'yuv420p', '-s',
         '640x240',
         DEMO_PATH])

    subprocess.call(['rm', '-rf', 'frames'])


def test(env, n_episode):
    total_rewards = 0

    for i_episode in range(n_episode):
        env.reset()

        for i in range(100):

            actions = env.action_space.sample()

            obs, rew, done, info = env.step(actions)

            total_rewards += np.sum(rew)

            if done:
                break

    avg_reward = total_rewards / (n_episode)

    print("Test ", n_episode, " episodes, average reward is: ", avg_reward)
    print("Test over.")

    return avg_reward


if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0.57, 0.35])
    high = np.array([0.63, 0.405])

    # obj_list = ['Milk', 'Bread', 'Cereal', 'Can', 'Banana', 'Bowl']
    # obj_num = [0, 0, 0, 0, 1, 1]
    # obj_names = []
    # for name, num in zip(obj_list, obj_num):
    #     obj_names += [name] * num

    render_drop_freq = 0

    env = suite.make(
        'BinPackPlace',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=1,

        camera_height=320,
        camera_width=240,

        render_drop_freq=render_drop_freq,
        # obj_names=obj_names,
        action_bound=(low, high),
    )

    ## make video
    n_episode = 2
    make_video(env, n_episode)

    ## test results
    n_episode = 30 * 8
    test(env, n_episode)
