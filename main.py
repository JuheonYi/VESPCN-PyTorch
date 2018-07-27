import torch

import data
import model
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)

loader = data.Data(args)
model = model.Model(args)
t = Trainer(args, loader, model)
while not t.terminate():
    t.train()
    t.test()

