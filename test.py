# Below are imports just for testing
from option import args
import cv2, imageio
from data.vsrdata import VSRData
"""
Original test setting: 5 videos with 30 frames inside each directory
length of frame sequence = 4
batch_size = 2
"""
'''
if __name__ == '__main__':
    if args.template == 'SY':
        vsr = VSRData(args, name='CDVL_Video', train=False)
    else:
        vsr = VSRData(args)
    print(len(vsr.data_hr))  # 5
    print(len(vsr.data_lr))  # 5
    print(vsr.data_hr[1].shape)  # (4,1080,1920,3)
    print(vsr.data_lr[1].shape)  # (4, 360, 640, 3)
    img_samples = []
    for i in range(args.n_sequence):
        imageio.imwrite('hr_{}.jpg'.format(i), vsr.data_hr[0][i, :])
        imageio.imwrite('lr_{}.jpg'.format(i), vsr.data_lr[0][i, :])
    print(len(vsr[0][0]))  # 4
    print(vsr[0][0][0].shape)  # torch.Size([3,17,17])
    print(len(vsr[0][1]))  # 4
    print(vsr[0][1][0].shape)  # torch.Size([3,51,51])
    print(vsr[0][2])  # ['00001', '00002', '00003', '00004']
'''

    
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from PIL import Image
import numpy as np

img = Image.open('./frame2.png')
img = np.array(img)
img = np.array([img]).astype("float64")
b = torch.from_numpy(img).double()
b = b.permute(0, 3, 1, 2)
print(b.size())
'''
flow = np.zeros((img.shape[1], img.shape[2], 2))
for i in range(0, img.shape[1]):
    for j in range(0, img.shape[2]):
        flow[i,j,:] = [i, j]
flow[:,:,0] = flow[:,:,0]/img.shape[1] 
flow[:,:,1] = flow[:,:,1]/img.shape[2]
flow = np.concatenate((flow[:,:,1:], flow[:,:,0:1]), axis=2)
flow = (flow - 0.5) * 2
flow = np.array([flow])
flow = torch.from_numpy(flow).double()
'''
# Create identity flow
x = np.linspace(-1, 1, img.shape[2]) - 0.01
y = np.linspace(-1, 1, img.shape[1]) - 0.01
xv, yv = np.meshgrid(x, y)
id_flow = np.expand_dims(np.stack([xv, yv], axis=-1), axis=0)
flow = torch.from_numpy(id_flow).double()

compensated = F.grid_sample(b, flow)
out = compensated.permute(0,2,3,1)
print(compensated.shape)
print(out.shape)
out = np.round(out.numpy()).astype("uint8")[0]
print(np.max(out), np.min(out))
imageio.imwrite("./out.png", out)