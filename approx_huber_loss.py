import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: Is the first channel flow with respect to x?
# TODO: Fix 'mean' issues


class Approx_Huber_Loss(nn.Module):
    def __init__(self, args):
        super(Approx_Huber_Loss, self).__init__()
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(self.device)
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(self.device)
        self.epsilon = torch.Tensor([0.01]).float().to(self.device)

    def forward(self, flow):
        flow_X = flow[:, 0:1]
        flow_Y = flow[:, 1:]
        grad_X = F.conv2d(flow_X, self.sobel_filter_X, bias=None, stride=1, padding=1)
        grad_Y = F.conv2d(flow_Y, self.sobel_filter_Y, bias=None, stride=1, padding=1)
        huber = torch.sqrt(self.epsilon + torch.sum(grad_X.pow(2)+grad_Y.pow(2)))
        return huber

