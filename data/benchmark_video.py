import os

from data import common
from data import srdata
from data import vsrdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark_video(vsrdata.VSRData):
    """
    Data generator for benchmark tasks
    """
    def __init__(self, args, name='', train=False):
        super(Benchmark_video, self).__init__(
            args, name=name, train=train
        )

    def _set_filesystem(self, dir_data):
        if self.args.template == 'SY':
            self.apath = os.path.join(dir_data, 'benchmark', self.name)
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic', 'X{}'.format(self.args.scale))
        ################################################
        #                                              #
        # Fill in your directory with your own template#
        #                                              #
        ################################################
        elif self.args.template == "JH_Video" or self.args.template == "JH_MC":
            self.apath = os.path.join(dir_data, self.name)
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR')
            print("test video path (HR):", self.dir_hr)
            print("test video path (LR):", self.dir_lr)

    def _load_file(self, idx):
        lr, hr, filename = super(Benchmark_video, self)._load_file(idx=idx)
        if self.name == 'Set14':
            if lr.ndim == 2:
                lr = np.repeat(np.expand_dims(lr, axis=2), 3, axis=2)
                hr = np.repeat(np.expand_dims(hr, axis=2), 3, axis=2)

        return lr, hr, filename
