import random
import numpy as np
import robosuite as suite
from baselines.ppo2 import ppo2

import os
import numpy as np
from PIL import Image
import subprocess

DEMO_PATH = 'temp'

if not os.path.exists(DEMO_PATH):
    os.makedirs(DEMO_PATH)

DEMO_PATH += '/test_demo.mp4'

def make_video(n_episode, slower=3):
    obj_names = (['Milk'] * 3 + ['Bread'] * 3 + ['Cereal'] * 3 + ['Can'] * 3)
    take_orders = ['Can1', 'Can2', 'Milk1', 'Milk2', 'Cereal1']

    low = np.array([0.57, 0.35])
    high = np.array([0.63, 0.405])

    render_drop_freq = 10

    env = suite.make(
        'BinSqueeze',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=False,
        use_camera_obs=True,
        control_freq=1,

        camera_height=240,
        camera_width=240,

        render_drop_freq=render_drop_freq,
        obj_names=obj_names,
        take_orders=take_orders,
        action_bound=(low, high),
    )

    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    time_step_counter = 0

    objects_palced = [
        np.array([0.57, 0.405]),
        np.array([0.63, 0.405]),
        np.array([0.558, 0.35]),
        np.array([0.642, 0.35]),
        # 'Cereal': np.array([0.60, 0.35, 1]),
    ]

    ## make video
    for i_episode in range(n_episode):
        env.reset()

        for _ in range(100):

            if _ < len(objects_palced):
                actions = objects_palced[_]
            else:
                actions = env.action_space.sample()

            obs, rew, done, info = env.step(actions)

            for i in range(len(info['birdview'])):
                image_data_bird, image_data_agent = info['birdview'][i], info['targetview'][i]
                image_data = np.concatenate((image_data_bird, image_data_agent), 1)

                img = Image.fromarray(image_data, 'RGB')
                for __ in range(slower):
                    img.save('frames/frame-%.10d.png' % time_step_counter)
                    time_step_counter += 1

            if done:
                break

    subprocess.call(
        ['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '24', '-pix_fmt', 'yuv420p', '-s',
         '480x240',
         DEMO_PATH])

    subprocess.call(['rm', '-rf', 'frames'])



if __name__ == "__main__":

    ## make video
    n_episode = 1
    slower = 1
    make_video(n_episode, slower)
