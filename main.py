import torch

import data
import model
from option import args
from trainer import Trainer
import os

torch.manual_seed(args.seed)

if not os.path.exists("./samples/{}".format(args.data_test)):
    os.makedirs("./samples/{}".format(args.data_test))


loader = data.Data(args)
model = model.Model(args)
t = Trainer(args, loader, model)
while not t.terminate():
    t.train()
    t.test()

