import os
import glob
import time

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.scale = args.scale
        self.idx_scale = 0

        data_range = [r.split('-') for r in args.data_range.split('/')]
        # Specify which data to read depending on training/testing
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_hr, self.images_lr = self._scan()
        if args.process:
            print('making numpy')
            self.data_hr, self.data_lr = self._load(self.images_hr, self.images_lr)

        if train:
            # self.repeat specifies how many epochs to iterate before evaluating
            self.repeat = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        """
        Returns a list of image directories
        """
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*.png')))
        names_lr = sorted(glob.glob(os.path.join(self.dir_lr, '*.png')))

        return names_hr, names_lr
    
    def _load(self, names_hr, names_lr):
        """

        :param names_hr: List of file paths to high-resolution images
        :param names_lr: List of file paths to low-resolution images
        :return: High-resolution and low-resolution data read from disk
        """
        data_lr = [imageio.imread(filename) for filename in names_lr]
        data_hr = [imageio.imread(filename) for filename in names_hr]
        
        return data_hr, data_lr

    def _set_filesystem(self, dir_data):
        """
        Specifies file directories for high-resolution and low-resolution data
        """
        self.apath = os.path.join(dir_data, self.name)
        if self.args.template == 'SY':
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
        elif self.args.template == "KJ":
            pass
            # TODO: specify data path

    def __getitem__(self, idx):
        if self.args.process:
            lr, hr, filename = self._load_file_from_loaded_data(idx)
        else:
            lr, hr, filename = self._load_file(idx)
        lr, hr = self.get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        """
        Prevent indices from going out of bounds
        """
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        """
        Read image at given image directory from disk
        """
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)

        return lr, hr, filename
    
    def _load_file_from_loaded_data(self, idx):
        """
        Read image directly from memory
        """
        idx = self._get_index(idx)
        hr = self.data_hr[idx]
        lr = self.data_lr[idx]
        filename = os.path.splitext(os.path.split(self.images_hr[idx])[-1])[0]

        return lr, hr, filename

    def get_patch(self, lr, hr):
        """
        Returns patches for multiple scales
        """
        scale = self.scale
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale  # TODO: What is idx_scale?
