import random
import argparse
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper
from baselines.ppo2 import ppo2
from robosuite.tests.load_model import load_model

import os
import numpy as np
from PIL import Image
import subprocess


def make_video(args, model, env, objects_palced, n_episode, slower=3):
    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    frame_dir = 'frames'
    time_step_counter = 0
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    num_env = args.num_env

    for i_episode in range(n_episode):
        obs = env.reset()

        import ipdb
        ipdb.set_trace()
        for pos in objects_palced:
            # pos = np.array([pos.tolist()] * num_env)
            obs, rew, done, info = env.step(pos)

        for _ in range(100):

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, info = env.step(actions)
            info = info[0]
            done = done[0]

            for i in range(len(info['birdview'])):
                image_data_bird, image_data_agent = info['birdview'][i], info['targetview'][i]
                image_data = np.concatenate((image_data_bird, image_data_agent), 1)

                img = Image.fromarray(image_data, 'RGB')
                for __ in range(slower):
                    img.save(frame_dir + '/frame-%.10d.png' % time_step_counter)
                    time_step_counter += 1

            if done:
                break

    subprocess.call(
        ['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '24', '-pix_fmt', 'yuv420p', '-s',
         '480x240',
         DEMO_PATH])

    subprocess.call(['rm', '-rf', 'frames'])


if __name__ == "__main__":

    ## params
    parser = argparse.ArgumentParser(description='Baseline Training...')

    ## env args
    parser.add_argument('--has_renderer', type=bool, default=False)
    parser.add_argument('--use_camera_obs', type=bool, default=False)
    parser.add_argument('--use_object_obs', type=bool, default=False)
    parser.add_argument('--has_offscreen_renderer', type=bool, default=False)
    parser.add_argument('--random_take', type=bool, default=False)

    parser.add_argument('--control_freq', type=int, default=1)
    parser.add_argument('--obj_nums', type=str, default='1,1,2,2')
    parser.add_argument('--obj_names', type=list, default=[])
    parser.add_argument('--camera_height', type=int, default=128)
    parser.add_argument('--camera_width', type=int, default=128)

    parser.add_argument('--keys', type=str, default='state', choices=['state', 'image'])
    parser.add_argument('--camera_name', type=str, default='targetview')

    ## alg args
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--num_env', type=int, default=4)
    parser.add_argument('--load_path', type=str, default='gg')
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--save_video_interval', type=int, default=0)

    parser.add_argument('--num_timesteps', type=int, default=200000)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--noptepochs', type=int, default=10)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--network', type=str, default='mlp')
    parser.add_argument('--num_layers', type=int, default=2)

    ## lr args
    parser.add_argument('--lr_type', type=str, default='const')
    parser.add_argument('--max', type=float, default=1e-3)
    parser.add_argument('--min', type=float, default=3e-4)

    ## video args
    parser.add_argument('--make_video', type=bool, default=False)
    parser.add_argument('--render_drop_freq', type=int, default=0)
    parser.add_argument('--video_name', type=str, default='demo.mp4')

    ## test args
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--test_episode', type=int, default=100)

    ## others
    parser.add_argument('--format_strs', type=list, default=['stdout', 'log', 'tensorboard'])
    parser.add_argument('--seed', default=None)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--debug', type=str, default='debug')

    args = parser.parse_args()

    ## const
    PATH = os.path.dirname(os.path.realpath(__file__))
    low = np.array([0.57, 0.35])
    high = np.array([0.63, 0.405])

    args.obj_types = ['Milk'] + ['Bread'] + ['Cereal'] + ['Can']
    args.take_orders = ['Can1', 'Can2', 'Milk1', 'Milk2', 'Cereal1']

    import ipdb
    ipdb.set_trace()

    model, env = load_model(args)

    # objects_palced = [
    #     np.array([0.57, 0.405, 0.91]),
    #     np.array([0.63, 0.405, 0.91]),
    #     np.array([0.558, 0.35, 0.88]),
    #     np.array([0.642, 0.35, 0.88]),
    #     # 'Cereal': np.array([0.60, 0.35, 1]),
    # ]
    objects_palced = [
        np.array([0.57, 0.405]),
        np.array([0.63, 0.405]),
        np.array([0.558, 0.35]),
        np.array([0.642, 0.35]),
        # 'Cereal': np.array([0.60, 0.35, 1]),
    ]

    ## make video
    n_episode = 1
    slower = 3
    make_video(args, model, env, objects_palced, n_episode, slower)