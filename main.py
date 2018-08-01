import torch

import data
import model
from option import args
from trainer import Trainer
from trainer_mc import Trainer_MC
from trainer_vsr import Trainer_VSR
from logger import logger

torch.manual_seed(args.seed)
chkp = logger.Logger(args)

if args.task == 'MC':
    model = model.Model(args, chkp)
    if args.load_all_videos:
        loader = data.Data(args)
        t = Trainer_MC(args, loader, model, chkp)
        while not t.terminate():
            t.train()
            t.test()

    else:
        loader = data.Data(args)
        t = Trainer_MC(args, loader, model, chkp)
        while not t.terminate():
            loader = data.Data(args)
            t.set_loader(loader)
            t.train()
            t.test()
elif args.task == 'Video':
    model = model.Model(args, chkp)
    if args.load_all_videos:
        loader = data.Data(args)
        t = Trainer_VSR(args, loader, model, chkp)
        while not t.terminate():
            t.train()
            #t.test()

    else:
        loader = data.Data(args)
        t = Trainer_VSR(args, loader, model, chkp)
        while not t.terminate():
            loader = data.Data(args)
            t.set_loader(loader)
            t.train()
            t.test()

elif args.task == 'Image':
    loader = data.Data(args)
    model = model.Model(args, chkp)
    t = Trainer(args, loader, model, chkp)
    while not t.terminate():
        t.train()
        t.test()

else:
    print('Please Enter Appropriate Task Type!!!')

chkp.done()
