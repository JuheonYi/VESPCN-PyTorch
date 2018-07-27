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
                filename = filename[0]
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)

                sr_img = utils.postprocess(sr.data.cpu().numpy()).astype('uint8')
                hr_img = utils.postprocess(hr.data.cpu().numpy()).astype('uint8')
                
                PSNR = utils.calc_PSNR(sr_img, hr_img)
                PSNR_avg += PSNR
                
                imageio.imwrite(("./samples/{}_SR.png".format(filename)), sr_img)
                imageio.imwrite(("./samples/{}_GT.png".format(filename)), hr_img)

            print("PSNR: %.3f" %(PSNR_avg/len(self.loader_test)))
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
