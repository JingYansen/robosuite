import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchvision import transforms
from PIL import Image

from robosuite.models.segmentation.pre_process import image_train, image_test
from robosuite.models.segmentation.mydataset import ImageList
from robosuite.models.segmentation.logger import Logger

## TODO; args, logger, train && test
## TODO: Divide dataset by type
def train(args):
    logger = Logger(ckpt_path=args.ckpt_path, tsbd_path=args.vis_path)

    models = []
    optimizers = []
    for i in range(4):
        model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005, nesterov=True)

        models.append(model)
        optimizers.append(optimizer)

    train_transforms = image_train()
    # test_transforms = image_test()

    train_dset = ImageList(open(args.train_list).readlines(), datadir=args.train_path, transform=train_transforms)
    # valid_dset = ImageList(open(args.valid_list).readlines(), datadir='', transform=valid_transforms)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    # valid_loader = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    total_epochs = 100
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=total_epochs * len(train_loader))

    for epoch in range(total_epochs):
        for it, (imgs, pixels, obj_types, rewards) in enumerate(train_loader):
            ## update log
            total_progress_bar.update(1)

            ## train the model
            for model in models:
                model.train()

            losses = 0.
            for img, pixel, obj_type, reward in zip(imgs, pixels, obj_types, rewards):
                img = img.cuda()
                pixel = pixel.cuda()
                reward = reward.cuda()
                model = models[obj_type]
                optimizer = optimizers[obj_type]

                img = img.unsqueeze(0)
                reward = reward.unsqueeze(0)
                mask = model(img)
                mask = mask[:, :, pixel[0], pixel[1]]
                loss = nn.CrossEntropyLoss()(mask, reward)
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ## vis
            logger.add_scalar('loss', losses / args.batch_size)

            if it % 100 == 0:
                print('loss: ', losses / args.batch_size)

        ## validate
        if epoch % args.test_interval == 1:
            ## validate
            # acc = validate(model, valid_loader)
            state = {}
            for i, model in enumerate(models):
                state['FPN_' + str(i)] = model.state_dict()
            logger.save_ckpt_iter(state=state, iter=epoch)

    state = {}
    for i, model in enumerate(models):
        state['FPN_' + str(i)] = model.state_dict()
    logger.save_ckpt_iter(state=state, iter=total_epochs)


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Segmentation Training...')

    parser.add_argument('--train_list', type=str, default='/home/yeweirui/data/temp/label.txt')
    parser.add_argument('--train_path', type=str, default='/home/yeweirui/')
    parser.add_argument('--ckpt_path', type=str, default='/home/yeweirui/temp')
    parser.add_argument('--vis_path', type=str, default='/home/yeweirui/temp')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_interval', type=int, default=10)

    args = parser.parse_args()

    train(args)