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
import cv2


class VSRData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale
        self.idx_scale = 0
        self.n_seq = args.n_sequence
        self.img_range = 30
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

        self.n_videos = len(self.images_hr)
        if train and args.process:
            self.data_hr, self.data_lr = self._load(self.n_videos)

        if train:
            self.repeat = args.test_every // (self.num_video // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        """
        Returns a list of image directories
        """
        if self.train:
            # training datasets are labeled as .../Video*/HR/*.png
            vid_hr_names = sorted(glob.glob(os.path.join(self.dir_hr, 'Video*')))
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr, 'Video*')))
        else:
            vid_hr_names = sorted(glob.glob(os.path.join(self.dir_hr,'*')))
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr,'*')))
            print("Number of test videos", len(vid_hr_names))

        self.num_video = len(vid_hr_names)
        assert len(vid_hr_names) == len(vid_lr_names)

        names_hr = []
        names_lr = []
        if self.args.load_all_videos == True:
            # Load videos all at once
            #for vid_hr_name, vid_lr_name in zip(vid_hr_names, vid_lr_names):
            for i in range(0,1):
                vid_hr_name = vid_hr_names[i]
                vid_lr_name = vid_lr_names[i]
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, '*.png')))
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, '*.png')))
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
        else:
            # If we do not want to load videos all at once, only load partial amount
            # TODO: load subset of videos (args.n_videos)
            for vid_hr_name, vid_lr_name in zip(vid_hr_names, vid_lr_names):
                start = self._get_index(random.randint(0, self.img_range - self.n_seq))
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, '*.png')))[start:start+self.n_seq]
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, '*.png')))[start:start+self.n_seq]
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
        return names_hr, names_lr

    def _load(self, n_videos):
        data_lr = []
        data_hr = []
        for idx in range(n_videos):
            lrs, hrs, _ = self._load_file(idx)
            data_lr.append(lrs)
            data_hr.append(hrs)
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
            print("apath:", self.apath)
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR')
        else:
            # This is just for testing: must fix later!
            self.dir_hr = os.path.join(self.apath, 'HR_big')
            self.dir_lr = os.path.join(self.apath, 'LR_big')

    def __getitem__(self, idx):
        if self.train and self.args.process:
            lrs, hrs, filenames = self._load_file_from_loaded_data(idx)
        else:
            lrs, hrs, filenames = self._load_file(idx)
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
            return idx % self.num_video
        else:
            return idx

    def _load_file(self, idx):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array and list of corresponding filenames
        """
        print("Loading video %d" %idx)
        f_hrs = self.images_hr[idx]
        f_lrs = self.images_lr[idx]

        filenames = [os.path.splitext(os.path.basename(file))[0] for file in f_hrs]
        hrs = np.array([imageio.imread(hr_name) for hr_name in f_hrs])
        lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs])
        #print(hrs.shape, lrs.shape)
        return lrs, hrs, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)
        idx = 0
        hrs = self.data_hr[idx]
        lrs = self.data_lr[idx]
        filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.images_hr[idx]]

        return lrs, hrs, filenames

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


