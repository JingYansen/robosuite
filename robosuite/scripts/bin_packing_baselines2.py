import sys
import os
import multiprocessing
import argparse
import copy
import cv2

import os.path as osp
import numpy as np
import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines import logger
from PIL import Image

from robosuite.scripts.utils import make_vec_env, norm_depth
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
    env_kwargs['camera_height'] = args.camera_height
    env_kwargs['camera_width'] = args.camera_width
    env_kwargs['use_camera_obs'] = args.use_camera_obs
    env_kwargs['has_renderer'] = args.has_renderer
    env_kwargs['has_offscreen_renderer'] = args.has_offscreen_renderer
    env_kwargs['camera_type'] = args.camera_type
    env_kwargs['random_take'] = args.random_take
    env_kwargs['take_nums'] = args.take_nums
    env_kwargs['use_typeVector'] = args.use_typeVector
    env_kwargs['make_dataset'] = args.make_dataset
    env_kwargs['dataset_path'] = args.dataset_path

    env_kwargs['keys'] = args.keys

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
                       flatten_dict_observations=flatten_dict_observations)
    return env


def make_video(model, env, args):
    DEMO_PATH = os.path.join(args.save_dir, args.video_name)

    import imageio
    writer = imageio.get_writer(DEMO_PATH, fps=20)

    n_episode = 10
    take_nums = args.take_nums

    ## set view
    # env.envs[0].render_drop_freq = 20
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    for i_episode in range(n_episode):
        obs = env.reset()
        total_reward = 0

        for _ in range(take_nums):

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards[0]

            for o in info[0]['birdview']:
                # contains depth

                image, depth = o
                depth = norm_depth(depth)

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                view = np.concatenate((image, depth), 0)

                text = str(total_reward)
                cv2.putText(view, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                writer.append_data(view)

            if dones[0]:
                break


    writer.close()
    print('Make video over.')
    print('Video path: ', DEMO_PATH)


def test(model, env, args):
    test_episode = args.test_episode
    num_env = args.num_env
    take_nums = args.take_nums
    avg_reward = 0

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((num_env,))

    logger.log('Begin testing, total ' + str(test_episode * num_env) + ' episodes...')
    for i_episode in range(test_episode):
        obs = env.reset()

        for _ in range(take_nums):
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rewards, dones, info = env.step(actions)
            avg_reward += np.sum(rewards)

    avg_reward /= (test_episode * num_env)
    logger.log('Average reward: ' + str(avg_reward))


def get_info_dir(args):
    info_dir = ''

    infos = [args.alg, args.network, args.lr_type, args.max, args.min]
    for info in infos:
        info_dir += str(info) + '_'

    keys = ['total', 'nsteps', 'noptepochs', 'batch', 'take', 'type', 'ent', 'dataset']
    values = [args.num_timesteps, args.nsteps, args.noptepochs, args.nminibatches, args.take_nums, args.use_typeVector, args.ent_coef, args.make_dataset]
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
    parser.add_argument('--use_object_obs', type=bool, default=True)
    parser.add_argument('--has_offscreen_renderer', type=bool, default=True)
    parser.add_argument('--camera_type', type=str, default='image+depth')
    parser.add_argument('--random_take', type=bool, default=True)
    parser.add_argument('--use_typeVector', type=bool, default=False)
    parser.add_argument('--make_dataset', type=bool, default=False)

    parser.add_argument('--control_freq', type=int, default=1)
    parser.add_argument('--render_drop_freq', type=int, default=0)
    parser.add_argument('--camera_height', type=int, default=128)
    parser.add_argument('--camera_width', type=int, default=128)
    parser.add_argument('--take_nums', type=int, default=6)

    parser.add_argument('--keys', type=str, default='image', choices=['state', 'image'])
    parser.add_argument('--dataset_path', type=str, default='data/temp/')

    ## alg args
    parser.add_argument('--env_id', type=str, default='BinPack-v0')
    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--num_env', type=int, default=16)
    parser.add_argument('--load_path', type=str, default='gg')

    parser.add_argument('--num_timesteps', type=int, default=1000000)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--noptepochs', type=int, default=10)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.2)
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--network', type=str, default='cnn')

    ## lr args
    parser.add_argument('--lr_type', type=str, default='const')
    parser.add_argument('--max', type=float, default=1e-3)
    parser.add_argument('--min', type=float, default=3e-4)

    ## video args
    parser.add_argument('--make_video', type=bool, default=False)
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
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])

    if args.use_typeVector:
        args.network = 'cnn_type'

    info_dir = get_info_dir(args)

    dir_list = [PATH, 'results', args.env_id, args.debug, args.camera_type, info_dir]

    args.save_dir = os.path.join(*dir_list)
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

    # import ipdb
    # ipdb.set_trace()
    model, env = train(args)

    if args.save_path is not None:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
        logger.log('Save to ', args.save_dir)

    if args.test:
        test(model, env, args)
        logger.log('Save to ', args.save_dir)

    if args.make_video:
        make_video(model, env, args)