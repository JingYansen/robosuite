import numpy as np
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchvision import transforms
from PIL import Image


def test(model, img):
    model = model.cuda()
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.cuda()

    mask = model(input_batch)[0]
    mask = mask.argmax(0)


    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(3)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color

    import ipdb
    ipdb.set_trace()

    r = Image.fromarray(mask.byte().cpu().numpy()).convert('RGB').resize((64, 64))
    r.putpalette(colors)

    r.save('gg.jpeg')


if __name__=='__main__':
    model = smp.FPN('resnet50', in_channels=4, classes=3)
    img = np.load('/home/yeweirui/data/temp/0/15983.npy')

    test(model, img)