import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        if train == True:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)
        self.images_hr, self.images_lr = self._scan()
        
        self.data_hr, self.data_lr = self._scan_and_load()
        print("images: ", len(self.data_hr), len(self.data_lr))
        print("size: ", self.data_hr[0].shape, self.data_lr[0].shape)

        if train:
            self.repeat = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        """
        Returns a list of image directories
        """
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + '.png')))
        names_lr = sorted(glob.glob(os.path.join(self.dir_lr, '*' + '.png')))
        #names_lr = [[] for _ in self.scale]
        #print(self.dir_lr)
        #print(names_lr)
        print("number of images:", len(names_lr))
        '''
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}.png'.format(s, filename)
                ))
        '''
        return names_hr, names_lr
    
    # Below functions as used to prepare images
    def _scan_and_load(self):
        """
        Returns loaded images in numpy array
        """
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + '.png')))
        names_lr = sorted(glob.glob(os.path.join(self.dir_lr, '*' + '.png')))

        data_hr = [imageio.imread(filename) for filename in names_hr]
        data_lr = [imageio.imread(filename) for filename in names_lr]
        
        return data_hr, data_lr

    def _set_filesystem(self, dir_data):

        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')

    def __getitem__(self, idx):
        #lr, hr, filename = self._load_file(idx)
        lr, hr, filename = self._load_file_from_loaded_data(idx)
        #lr, hr = self.get_patch(lr, hr)
        if self.train == True:
            lr, hr = common.get_patch(lr, hr, scale=3)
          
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        #print(lr.shape, hr.shape)
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )
        #print(lr_tensor.size(), hr_tensor.size())
        return lr_tensor, hr_tensor, filename

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

    def _load_file(self, idx):
        """
        Read image from given image directory
        """
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        #f_lr = self.images_lr[self.idx_scale][idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f:
                    hr = np.load(_f)[0]['image']
                with open(f_lr, 'rb') as _f:
                    lr = np.load(_f)[0]['image']

        return lr, hr, filename
    
    def _load_file_from_loaded_data(self, idx):
        """
        Read image from given image directory
        """
        idx = self._get_index(idx)
        hr = self.data_hr[idx]
        lr = self.data_lr[idx]
        filename = ""
        '''
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        #f_lr = self.images_lr[self.idx_scale][idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f:
                    hr = np.load(_f)[0]['image']
                with open(f_lr, 'rb') as _f:
                    lr = np.load(_f)[0]['image']
        '''
        return lr, hr, filename

    def get_patch(self, lr, hr):
        """
        Returns patches for multiple scales
        """
        #scale = self.scale[self.idx_scale]
        scale = self.scale
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
