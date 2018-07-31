import torch

import data
import model
from option import args
from trainer_vsr import Trainer_VSR
from logger import logger

torch.manual_seed(args.seed)
#chkp = logger.Logger(args)

print("VSR-main")
loader = data.Data(args)
model = model.Model(args, chkp)
t = Trainer_VSR(args, loader, model, chkp)
while not t.terminate():
    t.train()
    t.test()
chkp.done()