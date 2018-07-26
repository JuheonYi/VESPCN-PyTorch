import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

mean_RGB = np.array([123.68 ,  116.779,  103.939])

def preprocess(img):
    return (img - mean_RGB)/255 

def postprocess(img):
    return np.round(np.clip(img*255 + mean_RGB, 0, 255)).astype(np.uint8)

def calc_PSNR(img1, img2):
    #assume RGB image
    target_data = np.array(img1, dtype=np.float64)
    ref_data = np.array(img2,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)