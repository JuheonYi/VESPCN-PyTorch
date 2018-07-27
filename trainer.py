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
from utils import *

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
        #prev_time = time.time()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            #print("iteration:", time.time()-prev_time)
            #prev_time = time.time()
            #print(lr.size(), hr.size())
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
            #start = time.time()
            sr = self.model(lr)
            #print("inference:", time.time()-start)
            #print(lr.size(), sr.size(), hr.size())
            loss = self.loss(sr, hr)
            loss.backward()
            self.optimizer.step()
            
            if batch % 100 == 0:
                print("Iteration %d, loss: %.6f" %(batch, loss))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            '''
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                    sr = self.model(lr)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])
            '''
            eval_acc = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            PSNR_avg = 0
            for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                filename = filename[0]
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)
                
                PSNR = calc_PSNR(self.postprocess(sr.data.cpu().numpy()), self.postprocess(hr.data.cpu().numpy()))
                PSNR_avg += PSNR
                
                imageio.imwrite(("./samples/%d-SR.png" %idx_img), self.postprocess(sr.data.cpu().numpy()).astype('uint8'))
                imageio.imwrite(("./samples/%d-GT.png" %idx_img), self.postprocess(hr.data.cpu().numpy()).astype('uint8'))
                save_list = [sr]
                #if not no_eval:
                #    eval_acc += utility.calc_psnr(
                #            sr, hr, scale, self.args.rgb_range,
                #            benchmark=self.loader_test.dataset.benchmark
                #    )
                #    save_list.extend([lr, hr])
            print("PSNR: %.3f" %(PSNR_avg/idx_img))
    def postprocess(self, img):
        if img.shape[2] == 3:
            mean_RGB = np.array([123.68 ,  116.779,  103.939])
            out = img.squeeze(axis = 0).transpose((1, 2, 0))
            out = np.round(np.clip(out*255 + mean_RGB, 0, 255))
        else:
            mean_YCbCr = np.array([109])
            out = img.squeeze(axis = 0).transpose((1, 2, 0))
            out = np.round(np.clip(out*255 + mean_YCbCr, 0, 255))
        return out
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
