import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

"""
Repository for common functions required for manipulating data
"""


def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = sc.rgb2ycbcr(img)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        # NHWC -> NCHW
        if img.shape[2] == 3 and n_colors == 3:
            mean_RGB = np.array([123.68, 116.779, 103.939])
            img = img.astype('float64') - mean_RGB
        elif img.shape[2] == 3 and n_colors == 1:
            mean_YCbCr = np.array([109, 0, 0])
            img = img.astype('float64') - mean_YCbCr
        else:
            mean_YCbCr = np.array([109])
            img = img.astype('float64') - mean_YCbCr

        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = np.rot90(img)
        
        return img

    return [_augment(a) for a in args]

