import numpy as np


def norm_depth(depth, depth_max=0.99, depth_min=0.85):
    image = (depth - depth_min) / (depth_max - depth_min)
    image = np.clip(image, 0, 1)
    image = np.uint8(image * 255)

    image_shape = image.shape
    image = image.reshape(image_shape[0], image_shape[1], 1)

    return image