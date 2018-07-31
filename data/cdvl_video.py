import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

# Data loader for CDVL videos
class CDVL_VIDEO(vsrdata.VSRData):
    def __init__(self, args, name='CDVL', train=True):
        super(CDVL, self).__init__(
            args, name=name, train=train
        )
        
    def _set_filesystem(self, dir_data):
        ################################################
        #                                              #
        # Fill in your directory with your own template#
        #                                              #
        ################################################
        if self.args.template == "SY":
            super(CDVL100, self)._set_filesystem(dir_data)

        if self.args.template == "JH":
            print("Loading CDVL videos")
            self.apath = os.path.join(dir_data, self.name)
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR')
