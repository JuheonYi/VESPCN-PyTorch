import os
import math
import time
import imageio
import decimal

import numpy as np
from scipy import misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from tqdm import tqdm

import utils


class Trainer_VSR:
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp
        self.loss = nn.MSELoss()

    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        print("VSR training")
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))

        self.model.train()
        self.ckp.start_log()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            #print(lr.shape, hr.shape) #lr: [batch_size, n_seq, 3, patch_size, patch_size]
            if self.args.n_colors == 1 and lr.size()[2] == 3:
                lr = lr[:, :, 0:1, :, :]
                hr = hr[:, :, 0:1, :, :]

            # Divide LR frame sequence [N, n_sequence, n_colors, H, W] -> n_sequence * [N, 1, n_colors, H, W]    
            lr_frames = torch.split(lr, self.args.n_colors, dim = 1)
            # squeeze frames n_sequence * [N, 1, n_colors, H, W] -> n_sequence * [N, n_colors, H, W]
            lr_frames_squeezed = [torch.squeeze(frame, dim = 1) for frame in lr_frames]
            # concatenate frames n_sequence * [N, n_colors, H, W] -> [N, n_sequence * n_colors, H, W]
            lr_frames_cat = torch.cat(lr_frames_squeezed, dim = 1)
            
            # input frames = concatenated LR frames [N, n_sequence * n_colors, H, W]
            lr = lr_frames_cat
            # target frame = middle HR frame [N, n_colors, H, W]
            hr = hr[:, int(hr.shape[1]/2), : ,: ,:] 
            
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
            # output frame = single HR frame [N, n_colors, H, W] 
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            
            self.ckp.report_log(loss.item())
            loss.backward()
            self.optimizer.step()

            if batch % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {:.5f}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1)))

        self.ckp.end_log(len(self.loader_train))

    def test(self):
        def _torch_imresize(ttensor, scale):
            nparray = ttensor.data.cpu().numpy()
            nparray = np.transpose(np.squeeze(nparray, axis=0), (1,2,0))
            nparray = misc.imresize(nparray, size=scale*100, interp='bicubic')
            nparray = np.expand_dims(np.transpose(nparray, (2,0,1)), axis=0)
            return torch.from_numpy(nparray).float()

        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                #print(lr.shape)
                #print(hr.shape)
                ycbcr_flag = False
                filename = filename[0][0]
                #lr: [batch_size, n_seq, 3, patch_size, patch_size]
                if self.args.n_colors == 1 and lr.size()[2] == 3:
                    #print("converting to YCbCr")
                    # If n_colors is 1, split image into Y,Cb,Cr
                    ycbcr_flag = True
                    # for CbCr, select the middle frame
                    lr_center_y = lr[:, int(hr.shape[1]/2), 0:1, :, :].to(self.device)
                    lr_cbcr = lr[:, int(hr.shape[1]/2), 1:, :, :].to(self.device)
                    hr_cbcr = hr[:, int(hr.shape[1]/2), 1: ,: ,:].to(self.device)
                    sr_cbcr = _torch_imresize(lr[:, int(hr.shape[1]/2), :, :, :], self.args.scale)[:, 1:, :, :].to(self.device)
                    #print("cbcr shape", lr_cbcr.shape, hr_cbcr.shape, sr_cbcr.shape)

                    # extract Y channels (lr should be group, hr should be the center frame)
                    lr = lr[:, :, 0:1, :, :]
                    hr = hr[:, int(hr.shape[1]/2), 0:1, :, :]
                    #print("Y shape", lr.shape, hr.shape)
                    
                # Divide LR frame sequence [N, n_sequence, n_colors, H, W] -> n_sequence * [N, 1, n_colors, H, W]    
                lr_frames = torch.split(lr, self.args.n_colors, dim = 1)
                # squeeze frames n_sequence * [N, 1, n_colors, H, W] -> n_sequence * [N, n_colors, H, W]
                lr_frames_squeezed = [torch.squeeze(frame, dim = 1) for frame in lr_frames]
                # concatenate frames n_sequence * [N, n_colors, H, W] -> [N, n_sequence * n_colors, H, W]
                lr_frames_cat = torch.cat(lr_frames_squeezed, dim = 1)
            
                # input frames = concatenated LR frames [N, n_sequence * n_colors, H, W]
                lr = lr_frames_cat  
                #print("Y shape:", lr.shape)
                      
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)
                PSNR = utils.calc_psnr(self.args, sr, hr)
                self.ckp.report_log(PSNR, train=False)
                lr, lr_center_y, hr, sr = utils.postprocess(lr, lr_center_y, hr, sr,
                                               rgb_range=self.args.rgb_range,
                                               ycbcr_flag=ycbcr_flag, device=self.device)

                if self.args.save_images and idx_img%10 == 0:
                    #print("saving image %d" %idx_img)
                    if ycbcr_flag:
                        lr = torch.cat((lr_center_y, lr_cbcr), dim=1)
                        hr = torch.cat((hr, hr_cbcr), dim=1)
                        sr = torch.cat((sr, hr_cbcr), dim=1)

                    save_list = [lr, hr, sr]

                    self.ckp.save_images(filename, save_list, self.args.scale)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                self.args.data_test, self.ckp.psnr_log[-1],
                                best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs