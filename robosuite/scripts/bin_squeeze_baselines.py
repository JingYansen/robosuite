import sys
import os
import multiprocessing
import argparse
import copy

import os.path as osp
import numpy as np
import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines import logger
from PIL import Image

from robosuite.scripts.utils import make_vec_env
from robosuite.scripts.lr_schedule import get_lr_func
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_lr_kwargs(args):
    lr_kwargs = {}

    lr_kwargs['type'] = args.lr_type
    lr_kwargs['max'] = args.max
    lr_kwargs['min'] = args.min

    return lr_kwargs


def get_env_kwargs(args):
    env_kwargs = {}

    env_kwargs['render_drop_freq'] = args.render_drop_freq
    env_kwargs['control_freq'] = args.control_freq

    ## TODO: fix NoneType error in param obj_names.
    # env_kwargs['obj_names'] = args.obj_names
    env_kwargs['camera_height'] = args.camera_height
    env_kwargs['camera_width'] = args.camera_width
    env_kwargs['use_camera_obs'] = args.use_camera_obs
    env_kwargs['use_object_obs'] = args.use_object_obs
    env_kwargs['has_renderer'] = args.has_renderer
    env_kwargs['has_offscreen_renderer'] = args.has_offscreen_renderer
    env_kwargs['random_take'] = args.random_take

    if args.keys == 'state':
        env_kwargs['keys'] = 'state'
    elif args.keys == 'image':
        env_kwargs['keys'] = ['image']

    env_kwargs['camera_name'] = args.camera_name

    return env_kwargs

def get_params(args):
    params = {}

    params['nsteps'] = args.nsteps
    params['nminibatches'] = args.nminibatches
    params['noptepochs'] = args.noptepochs
    params['cliprange'] = args.cliprange
    params['ent_coef'] = args.ent_coef
    params['save_interval'] = args.save_interval
    params['log_interval'] = args.log_interval
    params['save_interval'] = args.save_interval
    params['network'] = args.network

    lr_kwargs = get_lr_kwargs(args)
    params['lr'] = get_lr_func(**lr_kwargs)

    if osp.exists(args.load_path):
        params['load_path'] = args.load_path
    else:
        logger.log('Warning: path <' + args.load_path + '> not exists.')

    return params

def get_network_params(args):
    params = {}

    if args.network is 'mlp':
        params['num_layers'] = args.num_layers

    return params


def train(args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    args.env_id, args.env_type = 'BinSqueeze-v0', 'mujoco'

    # get params
    learn = get_learn_function(args.alg)
    alg_kwargs = get_params(args)
    extra_args = get_network_params(args)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    alg_kwargs['network'] = args.network

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, args.env_type, args.env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    logger.log('Trained Over.')
    return model, env


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path, **kwargs)
    else:
        logger.configure(**kwargs)


def build_env(args):
    # make env in robosuite
    obj_names = []
    args.obj_nums = args.obj_nums.split(',')
    for i, name in zip(args.obj_nums, args.obj_types):
        obj_names = obj_names + [name] * int(i)

    args.obj_names = obj_names
    logger.log('Total objects: ', args.obj_names)

    # make env in gym
    env_kwargs = get_env_kwargs(args)

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu

    seed = args.seed

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = args.alg not in {'her'}
    env = make_vec_env(args.env_id, args.env_type, args.num_env or 1, seed, env_kwargs=env_kwargs,
                       reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)
    return env


def make_video(model, env, args):
    DEMO_PATH = 'demo'
    DEMO_PATH = os.path.join(DEMO_PATH, args.video_name)

    import imageio
    writer = imageio.get_writer(DEMO_PATH, fps=20)

    n_episode = 3
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    for i_episode in range(n_episode):
        obs = env.reset()

        for _ in range(1000):

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, info = env.step(actions)
            writer.append_data(obs)

            if done:
                break

    writer.close()
    print('Make video over.')


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
    if args.random_take:
        info_dir = 'random_'
    else:
        info_dir = 'fix_'

    infos = [args.keys, args.alg, args.network, args.lr_type, args.max, args.min]
    for info in infos:
        info_dir += str(info) + '_'

    keys = ['total', 'nsteps', 'env', 'clip', 'ent-coef', 'noptepochs', 'batch']
    values = [args.num_timesteps, args.nsteps, args.num_env, args.cliprange, args.ent_coef, args.noptepochs,
              args.nminibatches]
    assert len(keys) == len(values)

    for key, value in zip(keys, values):
        info_dir += str(value) + key + '_'

    if args.use_camera_obs:
        info_dir += '_' + str(args.camera_width) + 'x' + str(args.camera_height)

    return info_dir


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
    parser.add_argument('--out_dir', type=str, default='results_squeeze')
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
    parser.add_argument('--make_video', type=bool, default=True)
    parser.add_argument('--video_name', type=str, default='demo.mp4')

    ## test args
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--test_episode', type=int, default=100)

    ## others
    parser.add_argument('--format_strs', type=list, default=['stdout', 'log', 'tensorboard'])
    parser.add_argument('--seed', default=None)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--debug', type=str, default='debug')

    args = parser.parse_args()

    ## const
    PATH = os.path.dirname(os.path.realpath(__file__))
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])

    args.obj_types = ['Milk'] + ['Bread'] + ['Cereal'] + ['Can']

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
        make_video(model, env, args)