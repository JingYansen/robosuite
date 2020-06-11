import argparse
import torch

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

from PIL import Image
from torchvision import transforms


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Segmentation Training...')

    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--img_path', type=str, default='/home/yeweirui/data/temp/0/1.npy')
    parser.add_argument('--model_path', type=str, default='results/checkpoint_3.pth')
    parser.add_argument('--vis_name', type=str, default='results/show.jpg')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help="device id to run")

    args = parser.parse_args()

    model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()

    gpus = args.gpu_ids.split(',')
    model = nn.DataParallel(model, device_ids=[int(_) for _ in gpus])

    img_ori = np.load(args.img_path)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['FPN_' + str(args.type)])
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    img = preprocess(img_ori)
    img = img.unsqueeze(0).cuda()

    mask = model(img)[0]
    mask = mask.argmax(0)

    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([2 - i for i in range(3)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 255]
    ]).astype('uint8')

    seg = Image.fromarray(mask.byte().cpu().numpy()).resize((64, 64))
    seg.putpalette(colors)
    seg = seg.convert('RGB')

    # import ipdb
    # ipdb.set_trace()

    img_ori = img_ori[:, :, :3]
    img_ori = Image.fromarray(img_ori)
    img_seg = Image.new(img_ori.mode, (128, 64))

    img_seg.paste(img_ori, box=(0, 0))
    img_seg.paste(seg, box=(64, 0))

    img_seg.save(args.vis_name)