import sys
import os
import multiprocessing
import argparse
import copy

import os.path as osp
import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import logger

from robosuite.scripts.lr_schedule import get_lr_func

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def get_lr_kwargs(args):
    lr_kwargs = {}

    lr_kwargs['type'] = args.lr_type
    lr_kwargs['max'] = args.max
    lr_kwargs['min'] = args.min

    return lr_kwargs


def get_env_kwargs(args):
    env_kwargs = {}

    env_kwargs['control_freq'] = args.control_freq
    env_kwargs['camera_height'] = args.camera_height
    env_kwargs['camera_width'] = args.camera_width
    env_kwargs['use_camera_obs'] = args.use_camera_obs
    env_kwargs['has_renderer'] = args.has_renderer
    env_kwargs['has_offscreen_renderer'] = args.has_offscreen_renderer
    env_kwargs['camera_depth'] = args.camera_depth
    env_kwargs['reward_shaping'] = args.reward_shaping

    env_kwargs['total_steps'] = args.total_steps
    env_kwargs['step_size'] = args.step_size
    env_kwargs['orientation_scale'] = args.orientation_scale
    env_kwargs['energy_tradeoff'] = args.energy_tradeoff

    # hard_case_train, hard_case_test = get_hard_cases()
    # env_kwargs['obj_poses'] = hard_case_train

    env_kwargs['keys'] = args.keys

    return env_kwargs

def get_params(args):
    params = {}

    params['n_steps'] = args.nsteps
    params['nminibatches'] = args.nminibatches
    params['noptepochs'] = args.noptepochs
    params['cliprange'] = args.cliprange
    params['ent_coef'] = args.ent_coef

    lr_kwargs = get_lr_kwargs(args)
    params['learning_rate'] = get_lr_func(**lr_kwargs)

    # params['tensorboard_log'] = osp.join(args.save_dir, 'vis')

    return params


def train(args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    args.env_id, args.env_type = 'BinPack-v0', 'mujoco'

    # get params
    alg_kwargs = get_params(args)

    env = build_env(args)

    model = PPO2(CnnPolicy, env, verbose=1, **alg_kwargs)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )

    logger.log('Trained Over.')
    return model, env


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path, **kwargs)
    else:
        logger.configure(**kwargs)


def build_env(args):
    # make env in gym
    env_kwargs = get_env_kwargs(args)

    env = make_vec_env('BinSqueeze-v0', n_envs=args.num_env, env_kwargs=env_kwargs)
    return env


def make_video(model_path, env, args):
    model = PPO2.load(model_path)
    DEMO_PATH = os.path.join(args.save_dir, args.video_name)

    import imageio
    writer = imageio.get_writer(DEMO_PATH, fps=20)

    n_episode = 5

    for i_episode in range(n_episode):
        obs = env.reset()

        for _ in range(1000):

            action, _states = model.predict(obs)
            print('action: ', action)

            obs, rewards, dones, info = env.step(action)
            writer.append_data(obs[0])

            if dones[0]:
                break

    writer.close()
    print('Make video over.')
    print('Video path: ', DEMO_PATH)


def test(model, env, args):
    logger.log("Test...")

    n_episode = args.test_episode
    num_env = args.num_env
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    total_rewards = 0

    for i_episode in range(n_episode):
        obs = env.reset()

        for i in range(100):

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, info = env.step(actions)

            for r in rew:
                total_rewards += np.sum(r)

            done = done[0]
            if done:
                break

    avg_reward = total_rewards / (n_episode * num_env)

    if args.log:
        logger.log("Path: ", args.save_dir)
        logger.log("Test ", n_episode, " episodes, average reward is: ", avg_reward)
        logger.log("Test over.")
    else:
        print("Test ", n_episode, " episodes, average reward is: ", avg_reward)
        print("Test over.")


def get_info_dir(args):
    info_dir = ''

    infos = [args.alg, args.network, args.lr_type, args.max, args.min]
    for info in infos:
        info_dir += str(info) + '_'

    keys = ['total', 'nsteps', 'env', 'noptepochs', 'batch', 'shapeRw', 'depth', 'energy']
    values = [args.num_timesteps, args.nsteps, args.num_env, args.noptepochs, args.nminibatches, args.reward_shaping, args.camera_depth, args.energy_tradeoff]
    assert len(keys) == len(values)

    for key, value in zip(keys, values):
        info_dir += str(value) + key + '_'


    info_dir += str(args.camera_width) + 'x' + str(args.camera_height)

    return info_dir


if __name__ == "__main__":

    ## params
    parser = argparse.ArgumentParser(description='Baseline Training...')

    ## env args
    parser.add_argument('--has_renderer', type=bool, default=False)
    parser.add_argument('--use_camera_obs', type=bool, default=True)
    parser.add_argument('--has_offscreen_renderer', type=bool, default=True)
    parser.add_argument('--camera_depth', type=bool, default=False)
    parser.add_argument('--reward_shaping', type=bool, default=False)

    parser.add_argument('--control_freq', type=int, default=20)
    parser.add_argument('--camera_height', type=int, default=64)
    parser.add_argument('--camera_width', type=int, default=64)

    parser.add_argument('--total_steps', type=int, default=200)
    parser.add_argument('--step_size', type=float, default=0.002)
    parser.add_argument('--orientation_scale', type=float, default=0.1)
    parser.add_argument('--energy_tradeoff', type=float, default=0.02)

    parser.add_argument('--keys', type=str, default='image', choices=['image'])

    ## alg args
    parser.add_argument('--out_dir', type=str, default='results_squeeze')
    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--num_env', type=int, default=8)
    parser.add_argument('--load_path', type=str, default='gg')
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--save_video_interval', type=int, default=0)

    parser.add_argument('--num_timesteps', type=int, default=1000000)
    parser.add_argument('--nsteps', type=int, default=1024)
    parser.add_argument('--nminibatches', type=int, default=4)
    parser.add_argument('--noptepochs', type=int, default=20)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.005)
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--network', type=str, default='cnn')

    ## lr args
    parser.add_argument('--lr_type', type=str, default='const')
    parser.add_argument('--max', type=float, default=1e-3)
    parser.add_argument('--min', type=float, default=3e-4)

    ## video args
    parser.add_argument('--make_video', type=bool, default=True)
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

    info_dir = get_info_dir(args)

    args.save_dir = os.path.join(PATH, args.out_dir, args.debug, info_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        ans = input('Path with same params exists, overwirte or not?(yes/no)')
        if ans != 'yes':
            exit(0)

    args.save_path = os.path.join(args.save_dir, 'model.pth')

    if args.log:
        print('rank: ', MPI.COMM_WORLD.Get_rank())
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            rank = 0
            configure_logger(args.save_dir, format_strs=args.format_strs)
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            configure_logger(args.save_dir, format_strs=[])

        ## log
        logger.log(args)

    model, env = train(args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
        logger.log('Save to ', args.save_dir)

    if args.test:
        test(model, env, args)

    if args.make_video:
        if osp.exists(args.load_path):
            model_path = args.load_path
        else:
            model_path = args.save_path
        make_video(model_path, env, args)