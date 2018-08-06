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
    print("Selected task: MC")
    model = model.Model(args, chkp)
    loader = data.Data(args)
    t = Trainer_MC(args, loader, model, chkp)
    while not t.terminate():
        t.train()
        t.test()

elif args.task == 'Video':
    print("Selected task: Video")
    model = model.Model(args, chkp)
    loader = data.Data(args)
    t = Trainer_VSR(args, loader, model, chkp)
    while not t.terminate():
        t.train()
        t.test()

elif args.task == 'Image':
    print("Selected task: Image")
    loader = data.Data(args)
    model = model.Model(args, chkp)
    t = Trainer(args, loader, model, chkp)
    while not t.terminate():
        t.train()
        t.test()

else:
    print('Please Enter Appropriate Task Type!!!')

chkp.done()
