import os
import glob
import time
import skimage.color as sc
from data import common
import pickle
import numpy as np
import imageio
import random
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
        self.n_seq = args.n_sequence
        self.img_range = 30  # Number of images saved per video
        data_range = [r.split('-') for r in args.data_range.split('/')]
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
        if train and args.process:
            self.data_hr, self.data_lr = self._load(self.images_hr, self.images_lr)

        if train:
            self.repeat = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        """
        Returns a list of image directories
        """
        vid_names = sorted(glob.glob(os.path.join(self.dir_hr,'Video*')))
        names_hr = []
        names_lr = []
        for vid_name in vid_names:
            names_hr.append(sorted(glob.glob(os.path.join(vid_name, '*.png'))))
            names_lr.append(sorted(glob.glob(os.path.join(vid_name, '*.png'))))
        return names_hr, names_lr

    def _load(self, names_hr, names_lr):
        # TODO: Modify _load
        data_lr = [imageio.imread(filename) for filename in names_lr]
        data_hr = [imageio.imread(filename) for filename in names_hr]
        return data_hr, data_lr

    def _set_filesystem(self, dir_data):
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

    def __getitem__(self, idx):
        if self.train and self.args.process:
            lrs, hrs, filenames = self._load_file_from_loaded_data(idx, self.n_seq)
        else:
            lrs, hrs, filenames = self._load_file(idx, self.n_seq)
        patches = [self.get_patch(lr, hr) for lr, hr in zip(lrs, hrs)]
        lrs = np.array([patch[0] for patch in patches])
        hrs = np.array([patch[1] for patch in patches])
        if self.train:
            lrs = np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
            hrs = np.array(common.set_channel(*hrs, n_channels=self.args.n_colors))

        lr_tensors = common.np2Tensor(
            *lrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors
        )
        hr_tensors = common.np2Tensor(
            *hrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors
        )
        return lr_tensors, hr_tensors, filenames

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx, n_seq):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array
        """

        idx = self._get_index(random.randint(0, self.img_range-n_seq))
        f_hr = self.images_hr[idx:idx+n_seq]
        f_lr = self.images_lr[idx:idx+n_seq]

        filenames = [os.path.splitext(os.path.basename(file))[0] for file in f_hr]

        hr = np.array([imageio.imread(hr_name) for hr_name in f_hr])
        lr = np.array([imageio.imread(lr_name) for lr_name in f_lr])


        return lr, hr, filenames

    def _load_file_from_loaded_data(self, idx, n_seq):
        # TODO: Modify _load_file_from_loaded_data
        idx = self._get_index(random.randint(0, self.img_range-n_seq))
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
