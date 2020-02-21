import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper
import time

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

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0, 0])
    high = np.array([0.5, 0.5])
    obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2) * 2

    env = MyGymWrapper(
        suite.make(
            'BinPackPlace',
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
            obj_names=obj_names
        ),
        action_bound=(low, high)
    )

    # import ipdb
    # ipdb.set_trace()
