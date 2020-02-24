import sys
import os
import time
import multiprocessing
import argparse
import gym

import os.path as osp
import numpy as np
import tensorflow as tf
import robosuite as suite

from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines import logger

from robosuite.scripts.utils import make_vec_env
from robosuite.wrappers import MyGymWrapper
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


def get_params(args):
    params = {}

    params['nsteps'] = args.nsteps
    params['nminibatches'] = args.nminibatches
    params['ent_coef'] = args.ent_coef
    params['save_interval'] = args.save_interval
    params['log_interval'] = args.log_interval
    params['save_interval'] = args.save_interval
    params['lr'] = args.lr
    params['network'] = args.network

    return params

def get_network_params(args):
    params = {}

    if args.network is 'mlp':
        params['num_layers'] = args.num_layers

    return params


def train(args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    args.env_id, args.env_type = 'BinPack-v0', 'mujoco'

    ## get params
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

    return model, env


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def build_env(args):
    ## make env in robosuite
    obj_names = []
    for i, name in zip(args.obj_nums, args.obj_types):
        obj_names = obj_names + [name] * i

    logger.log('Total objects: ', obj_names)

    ## make env in gym

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
    env = make_vec_env(args.env_id, args.env_type, args.num_env or 1, seed, reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)

    # env = VecNormalize(env, use_tf=True)

    return env


if __name__ == "__main__":

    ## params
    parser = argparse.ArgumentParser(description='Baseline Training...')

    parser.add_argument('--out_dir', type=str, default='results/baselines')
    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--num_env', type=int, default=4)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--control_freq', type=int, default=1)
    parser.add_argument('--load_path', type=str, default='gg')
    parser.add_argument('--obj_nums', type=list, default=[1, 1, 2, 2])
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--save_video_interval', type=int, default=0)
    parser.add_argument('--seed', default=None)

    parser.add_argument('--num_timesteps', type=int, default=200000)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--network', type=str, default='mlp')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--debug', type=str, default='test3')


    args = parser.parse_args()

    ## const
    PATH = os.path.dirname(os.path.realpath(__file__))
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])

    args.obj_types = ['Milk'] + ['Bread'] + ['Cereal'] + ['Can']

    info_dir = 'states_' + args.alg + '_' + args.network + '_' + str(args.num_layers) + 'layer_' +\
               str(args.lr) + 'lr_' + str(args.nsteps) + 'stpes_' + str(args.num_env) + 'async_' + str(args.ent_coef) + 'explore_' +\
               args.debug

    args.save_dir = os.path.join(PATH, args.out_dir, info_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_path = os.path.join(args.save_dir, 'model.pth')

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.save_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.save_dir, format_strs=[])

    ## log
    logger.log(args)

    model, env = train(args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    logger.log('Trained Over.')
    logger.log('Save to ', args.save_dir)