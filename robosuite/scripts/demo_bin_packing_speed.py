import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper

import numpy as np
import time
from PIL import Image
import subprocess


if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])

    # obj_names = ['Milk'] * 2
    obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2) * 2
    # obj_names = ['Milk'] * 1 + ['Bread'] * 1 + ['Cereal'] * 1 + ['Can'] * 1

    has_renderer = False

    env = MyGymWrapper(
        suite.make(
            'BinPackPlace',
            has_renderer=has_renderer,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
            obj_names=obj_names,
        ),
        action_bound=(low, high)
    )

    # env.viewer.set_camera(camera_id=1)

    print('model_timestep: ', env.model_timestep)


    meter = []
    for _ in range(20):
        env.reset()
        for _ in range(len(obj_names)):

            if has_renderer:
                for _ in range(200):
                    env.render()

            action = env.action_space.sample()

            time1 = time.time()

            observation, reward, done, info = env.step(action)

            time2 = time.time()
            meter.append(time2 - time1)
            print('step time: ', time2 - time1, ' second')

    meter = np.array(meter)
    print('Average: ', meter.mean())
    print('FPS: ', 1.0 / meter.mean())