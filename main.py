import torch
import data
import model
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)  # Specify seed

loader = data.Data(args)  # Create an instance of 'Data'
model = model.Model(args)  # Create an instance of 'Model'
t = Trainer(args, loader, model)  # Create an instance of 'Trainer'
while not t.terminate():
    t.train()
    t.test()

