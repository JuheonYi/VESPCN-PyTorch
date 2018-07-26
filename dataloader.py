import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_data():
    HR_train = [1]
    HR_valid = [1]
    LR_train = [1]
    LR_valid = [1]
    print('1')
    return HR_train, HR_valid, LR_train, LR_valid