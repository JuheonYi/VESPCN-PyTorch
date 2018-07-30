import torch

import data
import model
from option import args
from trainer import Trainer
from logger import logger

torch.manual_seed(args.seed)
chkp = logger.Logger(args)


loader = data.Data(args)
model = model.Model(args, chkp)
t = Trainer(args, loader, model, chkp)
while not t.terminate():
    t.train()
    t.test()
chkp.done()