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
import robosuite as suite

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
from robosuite.models.segmentation.mydataset import action2pixel


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def overlap_image(img1, img2):
    img = Image.blend(img1, img2, (math.sqrt(5) - 1) / 2)
    return img


def make_video(env, seg_models, args):
    DEMO_PATH = args.video_path

    import imageio
    writer = imageio.get_writer(DEMO_PATH, fps=20)

    n_episode = 1
    take_nums = 6

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
        actions = [np.array([0.6425, 0.4225]), np.array([0.5575, 0.4225]), np.array([0.6425, 0.3375]), np.array([0.5575, 0.3375])]

        for _ in range(take_nums):

            import random
            action = random.choice(actions)
            # action = env.action_space.sample()
            # action[0] -= np.random.rand() / 20
            # action[1] -= np.random.rand() / 20
            # action[0] += (0.6425 - 0.5575) / take_nums
            # action[1] += (0.4225 - 0.3375) / take_nums

            obs, rewards, dones, info = env.step(action)
            total_reward += rewards

            obj_tp = info['obj_type']
            point = action2pixel(action)

            for o in info['birdview']:
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

            if dones:
                break


    writer.close()
    print('Make video over.')
    print('Video path: ', DEMO_PATH)


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Baseline Training...')

    parser.add_argument('--seg_model_path', type=str, default='results/random_take_8obj/checkpoint_10.pth')
    parser.add_argument('--video_path', type=str, default='results/debug/demo.mp4')
    parser.add_argument('--gpu_ids', type=str, default='0', help="device id to run")

    args = parser.parse_args()

    env = suite.make(
        'BinPackPlace',
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=1,
        render_drop_freq=20,
        camera_height=64,
        camera_width=64,
        video_height=64,
        video_width=64,
        take_nums=6,
        random_take=True
    )

    seg_models = []
    test_loaders = []
    gpus = args.gpu_ids.split(',')
    ckpt = torch.load(args.seg_model_path)
    for i in range(4):
        seg_model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()
        seg_model = nn.DataParallel(seg_model, device_ids=[int(_) for _ in gpus])
        seg_model.load_state_dict(ckpt['FPN_' + str(i)])

        seg_models.append(seg_model)

    make_video(env, seg_models, args)