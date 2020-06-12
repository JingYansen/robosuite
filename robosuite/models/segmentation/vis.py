import argparse
import torch
import os

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from robosuite.models.segmentation.mydataset import ImageList
from robosuite.models.segmentation.pre_process import image_test


def vis(args, model, loader):
    model.eval()

    idx = 0
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 255]
    ]).astype('uint8')

    for it, (imgs, pixels, obj_types, rewards) in enumerate(loader):

        imgs = imgs.cuda()
        pixels = pixels.cuda()
        rewards = rewards.cuda()
        masks = model(imgs)

        for img_ori, mask, pixel, obj_type in zip(masks, imgs, pixels, obj_types):
            mask = mask.argmax(0)

            seg = Image.fromarray(mask.byte().cpu().numpy()).resize((64, 64))
            seg.putpalette(colors)
            seg = seg.convert('RGB')

            import ipdb
            ipdb.set_trace()

            img_ori = img_ori.byte().detach().cpu().numpy()
            img_ori = img_ori.transpose((1,2,0))
            img_ori = Image.fromarray(img_ori)
            img_seg = Image.new(img_ori.mode, (128, 64))

            img_seg.paste(img_ori, box=(0, 0))
            img_seg.paste(seg, box=(64, 0))

            img_seg.save(os.path.join(args.vis_path, str(obj_type) + '_' + str(idx) + '.jpg'))

            idx += 1

            if idx > args.total_num:
                return


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Segmentation Training...')

    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--total_num', type=int, default=20)
    parser.add_argument('--img_path', type=str, default='/home/yeweirui/data/temp/0/1.npy')
    parser.add_argument('--model_path', type=str, default='results/checkpoint_3.pth')
    parser.add_argument('--data_list_path', type=str, default='/home/yeweirui/data/temp')
    parser.add_argument('--data_path', type=str, default='/home/yeweirui/')
    parser.add_argument('--vis_path', type=str, default='results/vis')
    parser.add_argument('--vis_name', type=str, default='results/show.jpg')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help="device id to run")

    args = parser.parse_args()

    if not os.path.exists(args.vis_path):
        os.mkdir(args.vis_path)

    model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()

    gpus = args.gpu_ids.split(',')
    model = nn.DataParallel(model, device_ids=[int(_) for _ in gpus])

    # img_ori = np.load(args.img_path)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['FPN_' + str(args.type)])
    # model.eval()
    #
    # preprocess = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    #
    # img = preprocess(img_ori)
    # img = img.unsqueeze(0).cuda()
    #
    # mask = model(img)[0]
    # mask = mask.argmax(0)
    #
    # colors = np.array([
    #     [255, 0, 0],
    #     [0, 255, 0],
    #     [255, 255, 255]
    # ]).astype('uint8')
    #
    # seg = Image.fromarray(mask.byte().cpu().numpy()).resize((64, 64))
    # seg.putpalette(colors)
    # seg = seg.convert('RGB')
    #
    # img_ori = img_ori[:, :, :3]
    # img_ori = Image.fromarray(img_ori)
    # img_seg = Image.new(img_ori.mode, (128, 64))
    #
    # img_seg.paste(img_ori, box=(0, 0))
    # img_seg.paste(seg, box=(64, 0))
    #
    # img_seg.save(args.vis_name)

    test_transform = image_test()

    test_list = os.path.join(args.data_list_path, 'label_' + str(args.type) + '_test.txt')
    test_dset = ImageList(open(test_list).readlines(), datadir=args.data_path, transform=test_transform)

    test_loader = DataLoader(test_dset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)

    vis(args, model, test_loader)