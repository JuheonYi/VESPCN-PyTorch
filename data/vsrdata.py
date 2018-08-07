import sys
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
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)
        # self.image_range : need to make it flexible in the test area
        self.img_range = 30
        self.n_frames_video = []
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
        self.num_video = len(self.images_hr)
        print("Number of videos to load:", self.num_video)
        if train:
            self.repeat = args.test_every // max((self.num_video // self.args.batch_size), 1)
        if args.process:
            self.data_hr, self.data_lr = self._load(self.num_video)

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
            vid_hr_names = sorted(glob.glob(os.path.join(self.dir_hr, '*')))
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr, '*')))

        assert len(vid_hr_names) == len(vid_lr_names)

        names_hr = []
        names_lr = []

        if self.train:
            for vid_hr_name, vid_lr_name in zip(vid_hr_names, vid_lr_names):
                start = random.randint(0, self.img_range - self.args.n_frames_per_video)
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, '*.png')))[start: start+self.args.n_frames_per_video]
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, '*.png')))[start: start+self.args.n_frames_per_video]
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
                self.n_frames_video.append(len(hr_dir_names))
        else:
            for vid_hr_name, vid_lr_name in zip(vid_hr_names, vid_lr_names):
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, '*.png')))
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, '*.png')))
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
                self.n_frames_video.append(len(hr_dir_names))

        return names_hr, names_lr

    def _load(self, n_videos):
        data_lr = []
        data_hr = []
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" %idx)
            lrs, hrs, _ = self._load_file(idx)
            hrs = np.array([imageio.imread(hr_name) for hr_name in self.images_hr[idx]])
            lrs = np.array([imageio.imread(lr_name) for lr_name in self.images_lr[idx]])
            data_lr.append(lrs)
            data_hr.append(hrs)
        #data_lr = common.set_channel(*data_lr, n_channels=self.args.n_colors)
        #data_hr = common.set_channel(*data_hr, n_channels=self.args.n_colors)
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
        if self.args.process:
            lrs, hrs, filenames = self._load_file_from_loaded_data(idx)
        else:
            lrs, hrs, filenames = self._load_file(idx)

        patches = [self.get_patch(lr, hr) for lr, hr in zip(lrs, hrs)]
        lrs = np.array([patch[0] for patch in patches])
        hrs = np.array([patch[1] for patch in patches])
        lrs = np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
        hrs = np.array(common.set_channel(*hrs, n_channels=self.args.n_colors))
        lr_tensors = common.np2Tensor(*lrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        hr_tensors = common.np2Tensor(*hrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        return torch.stack(lr_tensors), torch.stack(hr_tensors), filenames

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            # if test, call all possible video sequence fragments
            return sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_video
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array and list of corresponding filenames
        """

        if self.train:
            f_hrs = self.images_hr[idx]
            f_lrs = self.images_lr[idx]
            start = self._get_index(random.randint(0, self.n_frames_video[idx] - self.n_seq))
            filenames = [os.path.splitext(os.path.basename(file))[0] for file in f_hrs[start:start+self.n_seq]]
            hrs = np.array([imageio.imread(hr_name) for hr_name in f_hrs[start:start+self.n_seq]])
            lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs[start:start+self.n_seq]])

        else:
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_hrs = self.images_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            f_lrs = self.images_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_hrs]
            hrs = np.array([imageio.imread(hr_name) for hr_name in f_hrs])
            lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs])
        return lrs, hrs, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        if self.train:
            start = self._get_index(random.randint(0, self.n_frames_video[idx] - self.n_seq))
            hrs = self.data_hr[idx][start:start+self.n_seq]
            lrs = self.data_lr[idx][start:start+self.n_seq]
            filenames = [os.path.splitext(os.path.split(name)[-1])[0] for name in self.images_hr[idx]]

        else:
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_hrs = self.images_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            hrs = self.data_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            lrs = self.data_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_hrs]

        return lrs, hrs, filenames

    def get_patch(self, lr, hr):
        """
        Returns patches for multiple scales
        """
        scale = self.scale
        if self.train:
            patch_size = self.args.patch_size - (self.args.patch_size % 4)
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=patch_size,
                scale=scale,
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            ih -= ih % 4
            iw -= iw % 4
            lr = lr[:ih, :iw]
            hr = hr[:ih * scale, :iw * scale]

        return lr, hr
