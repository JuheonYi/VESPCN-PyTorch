import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return ESPCN(args)

class ESPCN(nn.Module):
#upscale_factor -> args
    def __init__(self, args):
        super(ESPCN, self).__init__()
        print("scale:", args.scale)
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 5, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 3, padding = 2)
        self.conv3 = nn.Conv2d(32, args.scale ** 2, kernel_size = 3, padding = 2)
        self.pixel_shuffle = nn.PixelShuffle(args.scale)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.tanh(x)
        return x
