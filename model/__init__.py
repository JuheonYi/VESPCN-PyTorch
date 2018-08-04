import os
from importlib import import_module

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        #print('Making model...')
        self.args = args
        self.scale = args.scale
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.ckp = ckp

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.get_model(), file=ckp.log_file)
    
    def forward(self, *args):
        return self.model(*args)
    
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, is_best=False, filename=''):
        target = self.get_model()
        filename = 'model_{}'.format(filename)
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', '{}latest.pt'.format(filename))
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', '{}best.pt'.format(filename))
            )

    def load(self, apath, pre_train='.', resume=False, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )

        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_best.pt'),
                    **kwargs
                ),
                strict=False
            )
            
