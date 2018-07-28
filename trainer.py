import os
import math
from decimal import Decimal

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from tqdm import tqdm
import time

import imageio
import utils

import cv2
import numpy as np

class Trainer:
    def __init__(self, args, loader, my_model):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        
        self.loss = nn.MSELoss()

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        self.scheduler.step()
        self.model.train()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                print("Iteration %d, loss: %.6f" %(batch, loss))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            PSNR_avg = 0
            for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                if self.args.n_colors == 1: # If n_colors is 1, split image into Y,Cb,Cr
                    ycbcr_lr = lr.clone()
                    ycbcr_hr = hr.clone()
                    
                    lr = torch.split(ycbcr_lr, 1, dim=1)[0]
                    hr = torch.split(ycbcr_hr, 1, dim=1)[0]
                    
                    lr_Cb = torch.split(ycbcr_lr, 1, dim=1)[1]
                    hr_Cb = torch.split(ycbcr_hr, 1, dim=1)[1]
                    
                    lr_Cr = torch.split(ycbcr_lr, 1, dim=1)[2]
                    hr_Cr = torch.split(ycbcr_hr, 1, dim=1)[2]
                filename = filename[0]
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)

                sr_img = utils.postprocess(sr.data.cpu().numpy()).astype('uint8')
                hr_img = utils.postprocess(hr.data.cpu().numpy()).astype('uint8')
                
                PSNR = utils.calc_PSNR(sr_img, hr_img)
                PSNR_avg += PSNR
                
                # save output image
                if self.args.n_colors == 3:
                    imageio.imwrite(("./samples/{}_SR.png".format(filename)), sr_img)
                    imageio.imwrite(("./samples/{}_GT.png".format(filename)), hr_img)
                else:
                    # stack YCbCr
                    Y = utils.postprocess(sr.data.cpu().numpy()).astype('uint8')
                    Y_GT = utils.postprocess(hr.data.cpu().numpy()).astype('uint8')
                    Cb_GT = utils.postprocess(hr_Cb.data.cpu().numpy()).astype('uint8')
                    Cr_GT = utils.postprocess(hr_Cr.data.cpu().numpy()).astype('uint8')
                    img_ycbcr_sr = np.dstack((Y, Cr_GT, Cb_GT))
                    img_rgb_sr = cv2.cvtColor(img_ycbcr_sr, cv2.COLOR_YCrCb2RGB)
                    imageio.imwrite(("./samples/{}/{}_SR_Y.png".format(self.args.data_test, filename)), Y)
                    imageio.imwrite(("./samples/{}/{}_SR_RGB.png".format(self.args.data_test, filename)), img_rgb_sr)
                    
                    img_ycbcr_GT = np.dstack((Y_GT, Cr_GT, Cb_GT))
                    img_rgb_GT = cv2.cvtColor(img_ycbcr_GT, cv2.COLOR_YCrCb2RGB)
                    imageio.imwrite(("./samples/{}/{}_GT_Y.png".format(self.args.data_test, filename)), Y_GT)
                    imageio.imwrite(("./samples/{}/{}_GT_RGB.png".format(self.args.data_test, filename)), img_rgb_GT)

            print("PSNR: %.3f" %(PSNR_avg/len(self.loader_test)))
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
