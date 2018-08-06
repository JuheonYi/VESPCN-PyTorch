import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def postprocess(*images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            mean_YCbCr = torch.Tensor([109]).to(device)
            out = (img.mul(rgb_coefficient) + mean_YCbCr).clamp(16, 235).div(rgb_coefficient)
        elif img.shape[2] == 3:
            mean_RGB = torch.Tensor([123.68, 116.779, 103.939]).to(device)
            mean_RGB = mean_RGB.reshape([1, 3, 1, 1])
            out = (img.mul(rgb_coefficient) + mean_RGB).clamp(0, 255).round().div(rgb_coefficient)
        else:
            mean_YCbCr = torch.Tensor([109]).to(device)
            out = (img.mul(rgb_coefficient) + mean_YCbCr).clamp(0, 255).round()
            out.div_(rgb_coefficient)

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]

'''
def calc_PSNR(img1, img2):
    # assume RGB image
    target_data = np.array(img1, dtype=np.float64)
    ref_data = np.array(img2, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)
'''

def calc_psnr(args, x, y):
    if isinstance(x, torch.Tensor):
        diff = (x - y).data
        shave = 2 + args.scale
        valid = diff[:, :, shave:-shave, shave:-shave]
        if args.n_colors == 3:
            convert = valid.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            valid.mul_(convert).div_(256)
            valid = valid.sum(dim=1)
        mse = valid.div(args.rgb_range).pow(2).mean()
        if mse == 0:
            mse = 1e-10
        # print('mse :', mse)
        return -10 * math.log10(mse)

    elif isinstance(x, np.ndarray):
        diff = (x - y)
        if diff.ndim == 4:
            diff = np.transpose(np.squeeze(diff, axis=0), (1, 2, 0))
        shave = 2 + args.scale
        valid = diff[shave:-shave, shave:-shave, :]
        if args.n_colors == 3:
            valid[:, :, 0] *= 65.738
            valid[:, :, 1] *= 129.057
            valid[:, :, 2] *= 25.064
            valid = valid.sum(axis=2) / 256
        mse = (valid ** 2).mean()
        if mse == 0:
            mse = 1e-10
        # print('mse :', mse)
        return -10 * math.log10(mse)
