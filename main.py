import torch
import data
import model
from option import args
from trainer import Trainer
import os

torch.manual_seed(args.seed)  # Specify seed

loader = data.Data(args)  # Create an instance of 'Data'
model = model.Model(args)  # Create an instance of 'Model'
t = Trainer(args, loader, model)  # Create an instance of 'Trainer'
if not os.path.exists("./samples/{}".format(args.data_test)):
    os.makedirs("./samples/{}".format(args.data_test))


loader = data.Data(args)
model = model.Model(args)
t = Trainer(args, loader, model)
while not t.terminate():
    t.train()
    t.test()

