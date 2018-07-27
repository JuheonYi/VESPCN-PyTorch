import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    """
    Data generator for benchmark tasks
    """
    def __init__(self, args, name='', train=True):
        super(Benchmark, self).__init__(
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
        elif self.args.template == "JH":
            self.apath = os.path.join(dir_data, self.name)
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR')


