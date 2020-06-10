import tqdm
import argparse
import numpy as np
import torch
import os
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


def CE_pixel(masks, pixels, labels):
    loss = 0.
    loss_fn = nn.CrossEntropyLoss()
    for batch in range(masks.size(0)):
        mask = masks.narrow(0, batch, 1)
        pixel = pixels[batch]
        mask = mask[:, :, pixel[0], pixel[1]]
        label = labels.narrow(0, batch, 1)

        loss += loss_fn(mask, label)
    return loss / masks.size(0)


def test(args, models, test_loaders):
    logger = args.logger
    accs = []

    for tp in range(args.type):
        train_loader = test_loaders[tp]
        model = models[tp]

        model.eval()
        acc = 0
        total = 0
        for it, (imgs, pixels, obj_types, rewards) in enumerate(train_loader):
            imgs = imgs.cuda()
            pixels = pixels.cuda()
            rewards = rewards.cuda()
            masks = model(imgs)

            for batch in range(masks.size(0)):
                pixel = pixels[batch]
                mask = masks[batch][:, pixel[0], pixel[1]]
                mask = mask.argmax(0)

                if mask == rewards[batch]:
                    acc += 1
            total += masks.size(0)
        acc /= total
        accs.append(acc)

    return accs

def train(args):
    logger = Logger(ckpt_path=args.ckpt_path, tsbd_path=args.vis_path)
    args.logger = logger

    train_transforms = image_train()
    test_transforms = image_test()

    models = []
    optimizers = []
    train_loaders = []
    test_loaders = []
    gpus = args.gpu_ids.split(',')
    for i in range(args.type):
        model = smp.FPN('resnet50', in_channels=4, classes=3).cuda()
        model = nn.DataParallel(model, device_ids=[int(_) for _ in gpus])

        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005, nesterov=True)

        models.append(model)
        optimizers.append(optimizer)

        train_list = os.path.join(args.data_list_path, 'label_' + str(i) + '_train.txt')
        test_list = os.path.join(args.data_list_path, 'label_' + str(i) + '_test.txt')

        train_dset = ImageList(open(train_list).readlines(), datadir=args.data_path, transform=train_transforms)
        test_dset = ImageList(open(test_list).readlines(), datadir=args.data_path, transform=test_transforms)

        train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)


    total_epochs = args.total_epochs
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=total_epochs)

    for epoch in range(total_epochs):
        total_progress_bar.update(1)
        ## train $type$ model
        for tp in range(args.type):
            train_loader = train_loaders[tp]
            test_loader = test_loaders[tp]
            model = models[tp]
            optimizer = optimizers[tp]

            model.train()

            temp_progress_bar = tqdm.tqdm(desc='Train iter for type ' + str(tp), total=len(train_loader))
            for it, (imgs, pixels, obj_types, rewards) in enumerate(train_loader):
                temp_progress_bar.update(1)

                imgs = imgs.cuda()
                pixels = pixels.cuda()
                rewards = rewards.cuda()
                masks = model(imgs)

                ## TODO: better format
                loss = CE_pixel(masks, pixels, rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## vis
                logger.add_scalar('loss_' + str(tp), loss.item() / args.batch_size)

        ## test
        if epoch % args.test_interval == 1:
            accs = test(args, models, test_loaders)
            import ipdb
            ipdb.set_trace()
            state = {}
            for i in range(args.type):
                state['FPN_' + str(i)] = models[i].state_dict()
                logger.add_scalar_print('acc_' + str(i), accs[i])

            logger.save_ckpt_iter(state=state, iter=epoch)

    ## at last
    accs = test(args, models, test_loaders)
    state = {}
    for i in range(args.type):
        state['FPN_' + str(i)] = models[i].state_dict()
        logger.add_scalar_print('acc_' + str(i), accs[i])

    logger.save_ckpt_iter(state=state, iter=total_epochs)


if __name__=='__main__':
    ## params
    parser = argparse.ArgumentParser(description='Segmentation Training...')

    # data
    parser.add_argument('--data_list_path', type=str, default='/home/yeweirui/data/temp')
    parser.add_argument('--data_path', type=str, default='/home/yeweirui/')

    # log
    parser.add_argument('--ckpt_path', type=str, default='results/')
    parser.add_argument('--vis_path', type=str, default='results/')

    # alg
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--type', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help="device id to run")

    args = parser.parse_args()

    train(args)