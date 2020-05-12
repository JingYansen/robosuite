import os
import cv2
import argparse

import os.path as osp
import numpy as np


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


def get_env_kwargs(args):
    env_kwargs = {}

    env_kwargs['control_freq'] = args.control_freq
    env_kwargs['camera_height'] = args.camera_height
    env_kwargs['camera_width'] = args.camera_width
    env_kwargs['use_camera_obs'] = args.use_camera_obs
    env_kwargs['has_renderer'] = args.has_renderer
    env_kwargs['has_offscreen_renderer'] = args.has_offscreen_renderer
    env_kwargs['camera_type'] = args.camera_type
    env_kwargs['fix_rotation'] = args.fix_rotation
    env_kwargs['random_target'] = args.random_target

    env_kwargs['random_quat'] = args.random_quat
    env_kwargs['neg_ratio'] = args.neg_ratio
    env_kwargs['place_num'] = args.place_num
    env_kwargs['test_cases'] = args.test_cases

    env_kwargs['total_steps'] = args.total_steps
    env_kwargs['step_size'] = args.step_size
    env_kwargs['orientation_scale'] = args.orientation_scale
    env_kwargs['energy_tradeoff'] = args.energy_tradeoff


    env_kwargs['keys'] = args.keys

    return env_kwargs


def build_env(args):
    # make env in gym
    env_kwargs = get_env_kwargs(args)

    env = make_vec_env(args.env_id, n_envs=args.num_env, env_kwargs=env_kwargs)
    return env


def make_video_multiPolicy(args, env, policies):
    DEMO_PATH = args.video_path

    import imageio
    writer = imageio.get_writer(DEMO_PATH, fps=20)

    n_episode = 30
    acc = 0

    assert len(policies) <= args.place_num

    for i_episode in range(n_episode):
        obs = env.reset()

        arr_imgs = []
        succ = False
        policy_id = 0

        for _ in range(1001):
            # import ipdb
            # ipdb.set_trace()

            assert policy_id < len(policies)
            model = policies[policy_id]

            action, _states = model.predict(obs)
            # print('action: ', action)

            obs, rewards, dones, info = env.step(action)

            # next policy
            if info[0]['this_down']:
                policy_id += 1

            data = info[0]['vis']
            # contains depth
            if data.shape[-1] == 4:
                image = data[:, :, :-1]
                depth = data[:, :, -1]

                depth_shape = depth.shape
                depth = depth.reshape(depth_shape[0], depth_shape[1], 1)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

                data = np.concatenate((image, depth), 0)

            # writer.append_data(data)
            arr_imgs.append(data)

            if dones[0]:
                succ = (rewards[0] >= 10)
                break

        if succ:
            acc += 1
            text = 'Success'
            color = (0, 255, 0)
        else:
            text = 'Fail'
            color = (255, 0, 0)

        for img in arr_imgs:
            cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

            writer.append_data(img)


    writer.close()
    print('Make video over.')
    print('Video path: ', DEMO_PATH)
    acc /= n_episode
    print('Acc rate: ', acc)


if __name__=='__main__':

    ## params
    parser = argparse.ArgumentParser(description='Multi polilcy testing')

    ## env args
    parser.add_argument('--has_renderer', type=bool, default=False)
    parser.add_argument('--use_camera_obs', type=bool, default=True)
    parser.add_argument('--has_offscreen_renderer', type=bool, default=True)
    parser.add_argument('--camera_type', type=str, default='image+depth')
    parser.add_argument('--fix_rotation', type=bool, default=True)
    parser.add_argument('--random_quat', type=bool, default=False)
    parser.add_argument('--random_target', type=bool, default=False)
    parser.add_argument('--neg_ratio', type=float, default=10)

    parser.add_argument('--control_freq', type=int, default=20)
    parser.add_argument('--camera_height', type=int, default=64)
    parser.add_argument('--camera_width', type=int, default=64)

    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--step_size', type=float, default=0.002)
    parser.add_argument('--orientation_scale', type=float, default=0.1)
    parser.add_argument('--energy_tradeoff', type=float, default=0)
    parser.add_argument('--place_num', type=int, default=5)
    parser.add_argument('--test_cases', type=list, default=[])

    parser.add_argument('--keys', type=str, default='image', choices=['image'])

    ## lr args
    parser.add_argument('--lr_type', type=str, default='const')
    parser.add_argument('--max', type=float, default=1e-5)
    parser.add_argument('--min', type=float, default=1e-5)

    ## alg args
    parser.add_argument('--num_timesteps', type=int, default=0)
    parser.add_argument('--nsteps', type=int, default=1024)
    parser.add_argument('--nminibatches', type=int, default=64)
    parser.add_argument('--noptepochs', type=int, default=10)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.2)

    ## env
    parser.add_argument('--env_id', type=str, default='BinSqueezeMulti-v0')
    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--num_env', type=int, default=1)

    ## policy dir, use model_1.pth, model_2.pth, model_3.pth...
    parser.add_argument('--load_dir', type=str, default='results/MultiStage/multi_test_dir')
    # parser.add_argument('--video_path', type=str, default='results/MultiStage/multi_test_dir/multi_test.mp4')

    args = parser.parse_args()
    args.video_path = os.path.join(args.load_dir, 'demo.mp4')

    ## begin
    total_timesteps = int(args.num_timesteps)

    env = build_env(args)
    num = 5
    model_names = ['model_' + str(i) + '.pth' for i in range(num)]
    load_paths = [os.path.join(args.load_dir, name) for name in model_names]

    policies = []
    alg_kwargs = get_params(args)

    for i, path in enumerate(load_paths):
        policy = PPO2.load(path)
        policies.append(policy)

    make_video_multiPolicy(args, env, policies)