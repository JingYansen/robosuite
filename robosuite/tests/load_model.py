import sys
import os
import multiprocessing
import argparse

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
    env_kwargs['obj_names'] = args.obj_names
    env_kwargs['camera_height'] = args.camera_height
    env_kwargs['camera_width'] = args.camera_width
    env_kwargs['use_camera_obs'] = args.use_camera_obs
    env_kwargs['use_object_obs'] = args.use_object_obs
    env_kwargs['has_renderer'] = args.has_renderer
    env_kwargs['has_offscreen_renderer'] = args.has_offscreen_renderer
    env_kwargs['take_orders'] = args.take_orders
    env_kwargs['random_take'] = args.random_take

    if args.keys == 'state':
        env_kwargs['keys'] = 'state'
    elif args.keys == 'image':
        env_kwargs['keys'] = ['image', 'obj_taken']

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
    args.env_id, args.env_type = 'BinPack-v0', 'mujoco'

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


def load_model(args):

    args.num_timesteps = 0
    ## const
    model, env = train(args)
    return model, env