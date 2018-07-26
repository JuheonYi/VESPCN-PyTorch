import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

mean_RGB = np.array([123.68 ,  116.779,  103.939])

def preprocess(img):
    return (img - mean_RGB)/255 

def postprocess(img):
    return np.round(np.clip(img*255 + mean_RGB, 0, 255)).astype(np.uint8)