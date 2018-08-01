import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return MotionCompensator(args)

class MotionCompensator(nn.Module):
    def __init__(self, args):
        super(MotionCompensator, self).__init__()
        print("Creating Motion compensator")

        def _gconv(in_channels, out_channels, kernel_size=3, groups=1, stride=1, bias=True):
            return nn.Conv2d(in_channels*groups, out_channels*groups, kernel_size, groups=groups, stride=stride,
                             padding=(kernel_size // 2), bias=bias)

        # Coarse flow
        self.coarse_flow = [_gconv(2, 24, kernel_size=5, groups=args.n_colors, stride=2), nn.ReLU(inplace=True)]
        self.coarse_flow.extend([_gconv(24, 24, kernel_size=3, groups=args.n_colors), nn.ReLU(True)])
        self.coarse_flow.extend([_gconv(24, 24, kernel_size=5, groups=args.n_colors, stride=2), nn.ReLU(True)])
        self.coarse_flow.extend([_gconv(24, 24, kernel_size=3, groups=args.n_colors), nn.ReLU(True)])
        self.coarse_flow.extend([_gconv(24, 32, kernel_size=3, groups=args.n_colors), nn.Tanh()])
        self.coarse_flow.extend([nn.PixelShuffle(4)])

        self.C_flow = nn.Sequential(*self.coarse_flow)

        # Fine flow
        self.fine_flow = [_gconv(5, 24, kernel_size=5, groups=args.n_colors, stride=2), nn.ReLU(inplace=True)]
        for _ in range(3):
            self.fine_flow.extend([_gconv(24, 24, kernel_size=3, groups=args.n_colors), nn.ReLU(True)])
        self.fine_flow.extend([_gconv(24, 8, kernel_size=3, groups=args.n_colors), nn.Tanh()])
        self.fine_flow.extend([nn.PixelShuffle(2)])

        self.F_flow = nn.Sequential(*self.fine_flow)
        
        #self.conv = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)

    def forward(self, frame_1, frame_2):
        # Coarse flow
        coarse_in = torch.cat((frame_1, frame_2), dim=1)
        coarse_out = self.C_flow(coarse_in)
        
        frame_2_compensated_coarse = self.warp(frame_2, coarse_out)
        
        # Fine flow
        fine_in = torch.cat((frame_1, frame_2, frame_2_compensated_coarse, coarse_out), dim=1)
        fine_out = self.F_flow(fine_in)
        
        flow = coarse_out + fine_out
        frame_2_compensated = self.warp(frame_2, flow)

        return frame_2_compensated, flow

    def warp(self, img, flow):
        # https://discuss.pytorch.org/t/solved-how-to-do-the-interpolating-of-optical-flow/5019
        # permute flow N C H W -> N H W C
        img_compensated = F.grid_sample(img, flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
        return img_compensated
