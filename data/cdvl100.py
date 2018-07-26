import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class CDVL100(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(CDVL100, self).__init__(
            args, name=name, train=train, benchmark=True
        )
