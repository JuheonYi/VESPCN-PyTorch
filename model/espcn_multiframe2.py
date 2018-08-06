import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(args):
    return ESPCN_multiframe2(args)


class ESPCN_multiframe2(nn.Module):
    # Add Residual connection!
    def __init__(self, args):
        super(ESPCN_multiframe2, self).__init__()
        print("Creating ESPCN multiframe2 (x%d)" % args.scale)
        network = [nn.Conv2d(args.n_colors * args.n_sequence, 64, kernel_size=3, padding=1), nn.ReLU(True)]
        network.extend([nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True)])
        network.extend([nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(True)])
        network.extend([nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(True)])
        network.extend(
            [nn.Conv2d(32, args.n_colors * args.scale * args.scale, kernel_size=3, padding=1), nn.ReLU(True)])
        network.extend([nn.PixelShuffle(args.scale)])
        network.extend([nn.Conv2d(args.n_colors, args.n_colors, kernel_size=1, padding=0)])

        self.net = nn.Sequential(*network)

    def forward(self, x):
        return self.net(x)
