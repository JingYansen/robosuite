import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper

import numpy as np
from PIL import Image
import subprocess

class human_policy:
    def __init__(self, low, high, delta=[0.06, 0.06]):
        self.low = np.copy(low)
        self.high = np.copy(high)
        self.delta = delta
        self.index = 0
        self.actions = []

        self.prepare()

    def prepare(self):
        for i in range(4):
            for j in range(5):
                action = self.low + self.delta * np.array([i, j])

                self.actions.append(action)


    def step(self):
        action = self.actions[self.index]
        self.index += 1
        return np.array(action)

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0.6, 0.3])
    high = low
    # high = np.array([0.66, 0.3])

    print('low: ', low)
    print('high: ', high)
    # obj_names = ['Cereal'] * 16
    obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2)
    # obj_names = ['Milk'] * 1 + ['Bread'] * 1 + ['Cereal'] * 1 + ['Can'] * 1

    has_renderer = True

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

    for i_episode in range(4):
        observation = env.reset()
        human = human_policy(low=low, high=high)
        for i in range(20):

            if has_renderer:
                for _ in range(200):
                    env.render()

            # action = env.action_space.sample()
            action = human.step()
            # print('action: ', action)
            observation, reward, done, info = env.step(action)

            if done:
                if has_renderer:
                    for _ in range(200):
                        env.render()
                break
