import argparse
import torch
import os
import cv2
import tqdm

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import DataLoader

from robosuite.models.segmentation.mydataset import ImageList
from robosuite.models.segmentation.pre_process import image_test


def show_data(img_path, save_file):
    img_np = np.load(img_path)[:, :, :3]
    img_np = Image.fromarray(img_np)
    img_np.save(save_file)


def vis(args, model, loader):
    model.eval()

    idx = 0
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 255]
    ]).astype('uint8')

    total_progress_bar = tqdm.tqdm(desc='Train iter', total=args.total_num)

    for it, (imgs, pixels, obj_types, rewards, paths) in enumerate(loader):

        imgs = imgs.cuda()
        pixels = pixels.cuda()
        rewards = rewards.cuda()
        masks = model(imgs)

        for _, mask, pixel, obj_type, path in zip(imgs, masks, pixels, obj_types, paths):
            point = pixel.detach().cpu().numpy()

            mask = mask.argmax(0)

            seg = Image.fromarray(mask.byte().cpu().numpy())#.resize((64, 64))
            seg.putpalette(colors)
            seg = seg.convert('RGB')
            draw = ImageDraw.Draw(seg)
            draw.ellipse((point[0]-1, point[1]-1, point[0]+1, point[1]+1), fill=(255, 255, 255))

            img_np = np.load(path)[:, :, :3]
            img_np = Image.fromarray(img_np)
            img_seg = Image.new(img_np.mode, (128, 64))

            img_seg.paste(img_np, box=(0, 0))
            img_seg.paste(seg, box=(64, 0))

            if args.train_data:
                f_name= os.path.join(args.vis_path, 'train_' + str(obj_type.item()) + '_' + str(idx) + '.jpg')
            else:
                f_name = os.path.join(args.vis_path, 'test_' + str(obj_type.item()) + '_' + str(idx) + '.jpg')
            img_seg.save(f_name)

            idx += 1
            total_progress_bar.update(1)

            if idx > args.total_num:
                return


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Segmentation Training...')

    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--total_num', type=int, default=20)
    # parser.add_argument('--img_path', type=str, default='/home/yeweirui/data/temp/0/1.npy')
    parser.add_argument('--train_data', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='results/random_data/checkpoint_1.pth')
    parser.add_argument('--data_list_path', type=str, default='/home/yeweirui/data/random')
    parser.add_argument('--data_path', type=str, default='/home/yeweirui/')
    parser.add_argument('--vis_path', type=str, default='results/random_data/vis')
    # parser.add_argument('--vis_name', type=str, default='results/show.jpg')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help="device id to run")

    args = parser.parse_args()

    if not os.path.exists(args.vis_path):
        os.mkdir(args.vis_path)

    model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()

    gpus = args.gpu_ids.split(',')
    model = nn.DataParallel(model, device_ids=[int(_) for _ in gpus])

    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['FPN_' + str(args.type)])

    test_transform = image_test()

    if args.train_data:
        test_list = os.path.join(args.data_list_path, 'label_' + str(args.type) + '_train.txt')
    else:
        test_list = os.path.join(args.data_list_path, 'label_' + str(args.type) + '_test.txt')
    test_dset = ImageList(open(test_list).readlines(), datadir=args.data_path, transform=test_transform, show_path=True)

    test_loader = DataLoader(test_dset, batch_size=4, shuffle=True, num_workers=4, drop_last=False)

    vis(args, model, test_loader)