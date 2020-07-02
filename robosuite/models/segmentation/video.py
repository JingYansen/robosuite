import os
import cv2
import sys
import math
import torch
import argparse
import multiprocessing

import os.path as osp
import numpy as np
import tensorflow as tf
import torch.nn as nn

from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines import logger
from PIL import Image, ImageDraw

from robosuite.scripts.utils import make_vec_env, norm_depth
from robosuite.scripts.lr_schedule import get_lr_func
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import segmentation_models_pytorch as smp

from torchvision import transforms
from robosuite.models.segmentation.pre_process import image_train, image_test
from robosuite.models.segmentation.mydataset import ImageList
from robosuite.models.segmentation.logger import Logger
from robosuite.models.segmentation.mydataset import action2pixel


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def overlap_image(img1, img2):
    img = Image.blend(img1, img2, (math.sqrt(5) - 1) / 2)
    return img


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
    env_kwargs['video_height'] = args.video_height
    env_kwargs['video_width'] = args.video_width
    env_kwargs['use_camera_obs'] = args.use_camera_obs
    env_kwargs['has_renderer'] = args.has_renderer
    env_kwargs['has_offscreen_renderer'] = args.has_offscreen_renderer
    env_kwargs['camera_type'] = args.camera_type
    env_kwargs['random_take'] = args.random_take
    env_kwargs['take_nums'] = args.take_nums
    env_kwargs['use_typeVector'] = args.use_typeVector
    env_kwargs['make_dataset'] = args.make_dataset

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

    if osp.exists(args.rl_model_path):
        params['load_path'] = args.rl_model_path
    else:
        raise ValueError('No path for ', args.rl_model_path)

    return params


def get_network_params(args):
    params = {}

    if args.network is 'mlp':
        params['num_layers'] = args.num_layers

    return params


def build_env(args):
    # make env in gym
    env_kwargs = get_env_kwargs(args)

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu

    seed = None

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = args.alg not in {'her'}
    env = make_vec_env(args.env_id, args.env_type, args.num_env or 1, seed, env_kwargs=env_kwargs,
                       flatten_dict_observations=flatten_dict_observations)
    return env


def train(args):
    total_timesteps = int(args.num_timesteps)
    seed = None
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



def make_video(model, env, seg_models, args):
    DEMO_PATH = args.video_path

    import imageio
    writer = imageio.get_writer(DEMO_PATH, fps=20)

    n_episode = args.n_episode
    take_nums = args.take_nums

    ## set view
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 255]
    ]).astype('uint8')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

            obj_tp = info[0]['obj_type']
            point = action2pixel(actions[0])

            for o in info[0]['birdview']:
                # contains depth

                image, depth = o
                depth = norm_depth(depth)

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)
                # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                # get 4 channel obs
                view = np.concatenate((image, depth), 2)

                # seg
                seg_model = seg_models[obj_tp]
                seg_model.eval()

                input_tensor = preprocess(view)
                input_batch = input_tensor.unsqueeze(0).cuda()

                mask = seg_model(input_batch)[0]
                mask = mask.argmax(0)

                seg = Image.fromarray(mask.byte().cpu().numpy())
                seg.putpalette(colors)
                seg = seg.convert('RGB')

                # overlap
                seg = overlap_image(seg, Image.fromarray(image))

                draw = ImageDraw.Draw(seg)
                draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=(255, 255, 255))

                # get original image
                text = str(total_reward)
                cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

                img_np = Image.fromarray(image)
                img_seg = Image.new(img_np.mode, (128, 64))
                img_seg.paste(img_np, box=(0, 0))
                img_seg.paste(seg, box=(64, 0))

                data = np.asarray(img_seg)

                writer.append_data(data)

            if dones[0]:
                break


    writer.close()
    print('Make video over.')
    print('Video path: ', DEMO_PATH)


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Baseline Training...')

    ## env args
    parser.add_argument('--has_renderer', type=bool, default=False)
    parser.add_argument('--use_camera_obs', type=bool, default=True)
    parser.add_argument('--use_object_obs', type=bool, default=True)
    parser.add_argument('--has_offscreen_renderer', type=bool, default=True)
    parser.add_argument('--camera_type', type=str, default='image+depth')
    parser.add_argument('--random_take', type=bool, default=True)
    parser.add_argument('--use_typeVector', type=bool, default=True)
    parser.add_argument('--make_dataset', type=bool, default=False)

    parser.add_argument('--control_freq', type=int, default=1)
    parser.add_argument('--render_drop_freq', type=int, default=20)

    parser.add_argument('--video_height', type=int, default=64)
    parser.add_argument('--video_width', type=int, default=64)
    parser.add_argument('--camera_height', type=int, default=64)
    parser.add_argument('--camera_width', type=int, default=64)
    parser.add_argument('--take_nums', type=int, default=6)

    parser.add_argument('--keys', type=str, default='image', choices=['state', 'image'])

    ## alg args
    parser.add_argument('--env_id', type=str, default='BinPack-v0')
    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--rl_model_path', type=str, default='../../scripts/results/BinPack-v0/version-1.1.8/image+depth/ppo2_cnn_type_linear_0.0001_0.0001_2000000total_256nsteps_10noptepochs_32batch_6take_Truetype_0.005ent_Falsedataset_64x64/model.pth')

    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--noptepochs', type=int, default=10)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.005)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--network', type=str, default='cnn')

    ## lr args
    parser.add_argument('--lr_type', type=str, default='const')
    parser.add_argument('--max', type=float, default=1e-4)
    parser.add_argument('--min', type=float, default=1e-4)

    ## video args
    parser.add_argument('--n_episode', type=int, default=10)
    parser.add_argument('--video_path', type=str, default='results/test/demo.mp4')

    ## seg model
    parser.add_argument('--gpu_ids', type=str, default='0', help="device id to run")
    parser.add_argument('--seg_model_path', type=str, default='results/random_take_8obj_1e-4_ok/checkpoint_3.pth')

    args = parser.parse_args()

    if args.use_typeVector:
        args.network = 'cnn_type'
    print('Load RL model.')
    model, env = train(args)

    print('Load Segmentation model.')
    seg_models = []
    test_loaders = []
    gpus = args.gpu_ids.split(',')
    ckpt = torch.load(args.seg_model_path)
    for i in range(4):
        seg_model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()
        seg_model = nn.DataParallel(seg_model, device_ids=[int(_) for _ in gpus])
        seg_model.load_state_dict(ckpt['FPN_' + str(i)])

        seg_models.append(seg_model)

    make_video(model, env, seg_models, args)