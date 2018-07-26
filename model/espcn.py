import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return EDSR(args)

class ESPCN(nn.Module):
#upscale_factor -> args
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size = 5, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 3, padding = 2)
        self.conv3 = nn.Conv2d(32, upscale_factor ** 2, kernel_size = 3, padding = 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.tanh(x)
        return x
