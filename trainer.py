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


class Trainer:
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

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))

        self.model.train()
        self.ckp.start_log()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            if self.args.n_colors == 1 and lr.size()[1] == 3:
                lr = lr[:, 0:1, :, :]
                hr = hr[:, 0:1, :, :]

            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
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
                ycbcr_flag = False
                if self.args.n_colors == 1 and lr.size()[1] == 3:
                    print("converting to YCbCr")
                    # If n_colors is 1, split image into Y,Cb,Cr
                    ycbcr_flag = True
                    sr_cbcr = _torch_imresize(lr, self.args.scale)[:, 1:, :, :].to(self.device)
                    lr_cbcr = lr[:, 1:, :, :].to(self.device)
                    lr = lr[:, 0:1, :, :]
                    hr_cbcr = hr[:, 1:, :, :].to(self.device)
                    hr = hr[:, 0:1, :, :]

                filename = filename[0]
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)
                PSNR = utils.calc_psnr(self.args, sr, hr)
                self.ckp.report_log(PSNR, train=False)
                lr, hr, sr = utils.postprocess(lr, hr, sr,
                                               rgb_range=self.args.rgb_range,
                                               ycbcr_flag=ycbcr_flag, device=self.device)

                if ycbcr_flag:
                    lr = torch.cat((lr, lr_cbcr), dim=1)
                    hr = torch.cat((hr, hr_cbcr), dim=1)
                    sr = torch.cat((sr, hr_cbcr), dim=1)

                save_list = [lr, hr, sr]
                if self.args.save_images:
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