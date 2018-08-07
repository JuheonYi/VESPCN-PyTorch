import time
import decimal
import numpy as np

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from approx_huber_loss import Approx_Huber_Loss

import utils

class Trainer_MC:
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
        self.flow_loss = Approx_Huber_Loss(args)

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()

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
        for batch, (lr, _, _) in enumerate(self.loader_train):
            # tensor size of lr : B*n*C*H*W (H, W = args.patch_size)
            self.optimizer.zero_grad()
            if self.args.n_colors == 1 and lr.size()[-3] == 3:
                lr = lr[:, :, 0:1, :, :]
            lr = lr.to(self.device)
            frame1, frame2 = lr[:, 0], lr[:, 1]
            frame2_compensated, flow = self.model(frame1, frame2)
            loss = self.loss(frame2_compensated, frame1) + self.args.lambd * self.flow_loss(flow)
            
            self.ckp.report_log(loss.item())  # TODO: Check logging issues for Huber loss
            loss.backward()
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {:.5f}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1)))

        self.ckp.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (lr, _, filename) in enumerate(tqdm_test):
                ycbcr_flag = False
                filename = filename[0][0]
                lr = lr.to(self.device)
                frame1, frame2 = lr[:, 0], lr[:, 1]
                if self.args.n_colors == 1 and lr.size()[-3] == 3:
                    ycbcr_flag = True
                    frame1_cbcr = frame1[:, 1:]
                    frame2_cbcr = frame2[:, 1:]
                    frame1 = frame1[:, 0:1]
                    frame2 = frame2[:, 0:1]

                frame2_compensated, flow = self.model(frame1, frame2)

                PSNR = utils.calc_psnr(self.args, frame1, frame2_compensated)
                self.ckp.report_log(PSNR, train=False)
                frame1, frame2, frame2c = utils.postprocess(frame1, frame2, frame2_compensated,
                                            rgb_range=self.args.rgb_range, ycbcr_flag=ycbcr_flag, device=self.device)

                if ycbcr_flag:
                    frame1 = torch.cat((frame1, frame1_cbcr), dim=1)
                    frame2 = torch.cat((frame2, frame2_cbcr), dim=1)
                    frame2_cbcr_c = F.grid_sample(frame2_cbcr, flow.permute(0, 2, 3, 1), padding_mode='border')
                    frame2c = torch.cat((frame2c, frame2_cbcr_c), dim=1)

                save_list = [frame1, frame2, frame2c]
                if self.args.save_images and idx_img%10 == 0:
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
