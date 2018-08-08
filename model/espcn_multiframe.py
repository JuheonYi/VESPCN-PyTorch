import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return ESPCN_multiframe(args)

class ESPCN_multiframe(nn.Module):
#upscale_factor -> args
    def __init__(self, args):
        super(ESPCN_multiframe, self).__init__()
        self.name = 'ESPCN_mf'
        print("Creating ESPCN multiframe (x%d)" %args.scale)
        '''
        self.network = [nn.Conv2d(args.n_colors*args.n_sequence, 24, kernel_size = 3, padding =1), nn.ReLU(True)]
        for i in range(0,3):
            self.network.extend([nn.Conv2d(24, 24, kernel_size = 3, padding =1), nn.ReLU(True)])
        
        self.network.extend([nn.Conv2d(24, args.n_colors * args.scale * args.scale, kernel_size = 3, padding =1), nn.ReLU(True)])
        self.network.extend([nn.PixelShuffle(args.scale)])
        self.network.extend([nn.Conv2d(args.n_colors, args.n_colors, kernel_size = 1, padding = 0)])
        '''
        self.network = [nn.Conv2d(args.n_colors*args.n_sequence, 64, kernel_size = 5, padding =2), nn.ReLU(True)]
        self.network.extend([nn.Conv2d(64, 48, kernel_size = 3, padding =1), nn.ReLU(True)])
        self.network.extend([nn.Conv2d(48, 32, kernel_size = 3, padding =1), nn.ReLU(True)])
        self.network.extend([nn.Conv2d(32, 24, kernel_size = 3, padding =1), nn.ReLU(True)])
        self.network.extend([nn.Conv2d(24, args.n_colors * args.scale * args.scale, kernel_size = 3, padding =1), nn.ReLU(True)])
        self.network.extend([nn.PixelShuffle(args.scale)])
        self.network.extend([nn.Conv2d(args.n_colors, args.n_colors, kernel_size = 1, padding = 0)])
        
        self.net = nn.Sequential(*self.network)
       

    def forward(self, x):
        if isinstance(x, list):
            # squeeze frames n_sequence * [N, 1, n_colors, H, W] -> n_sequence * [N, n_colors, H, W]
            lr_frames_squeezed = [torch.squeeze(frame, dim = 1) for frame in x]
            # concatenate frames n_sequence * [N, n_colors, H, W] -> [N, n_sequence * n_colors, H, W]
            x = torch.cat(lr_frames_squeezed, dim = 1)    
        
        return self.net(x)