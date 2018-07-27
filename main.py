import numpy as np
import torch
import torch.optim as optim
from torch.autograd.variable import Variable
import time
import data
import model
from option import args
import torch.nn as nn

loader = data.Data(args)
net = model.Model(args)
print(net)

MSE_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

#net.load_state_dict(torch.load('./checkpoints/ESPCN.pkl'))
start_time = time.time()
for epoch in range(2000):
    if epoch % 100 == 0:
        print('Epoch %d, runtime: %.4f' %(epoch + 1, time.time()-start_time))
    start_time = time.time()
    
    for batch, (LR, HR, _, idx_scale) in enumerate(loader.loader_train):
        #print(i)
        HR = HR.type(torch.FloatTensor) 
        inputs, target = Variable(LR), Variable(HR)
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        
        outputs = net(inputs)
        #print("outputs:", outputs,"steering:",steering)
        
        loss = MSE_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        print(loss)
        #if epoch % 100 == 0 and i % 50 == 49:   
        #    print('[%d, %5d] loss : %.6f' % (epoch + 1, i+ 1, loss.data[0] / batch_num))
    
print('Finished training, runtime: %.4f' %(time.time()-start_time))

torch.save(net.state_dict(), './checkpoints/ESPCN.pkl')