import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return ESPCN_modified(args)

class ESPCN_modified(nn.Module):
#upscale_factor -> args
    def __init__(self, args):
        super(ESPCN_modified, self).__init__()
        print("Creating modified ESPCN (x%d)" %args.scale)
        self.conv1 = nn.Conv2d(args.n_colors, 64, kernel_size = 5, padding = 2)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(32, args.n_colors * args.scale * args.scale, kernel_size = 3, padding = 1)
        self.pixel_shuffle = nn.PixelShuffle(args.scale)
        self.conv4 = nn.Conv2d(args.n_colors, args.n_colors, kernel_size = 1, padding = 0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.relu(self.conv1_3(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        #x = self.tanh(x)
        x = self.conv4(x)
        return x
