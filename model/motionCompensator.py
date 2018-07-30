import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return MotionCompensator(args)

class MotionCompensator(nn.Module):
#upscale_factor -> args
    def __init__(self, args):
        super(MotionCompensator, self).__init__()
        print("Creating Motion compensator")
        
        # Coarse flow
        self.c_conv1 = nn.Conv2d(args.n_colors, 24, kernel_size = 5, stride = 2, padding = 2)
        self.c_conv2 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)
        self.c_conv3 = nn.Conv2d(24, 24, kernel_size = 5, stride = 2, padding = 2)
        self.c_conv4 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)
        self.c_conv5 = nn.Conv2d(24, 32, kernel_size = 3, padding = 1)
        self.c_pixel_shuffle = nn.PixelShuffle(4)
        
        # Fine flow
        self.f_conv1 = nn.Conv2d(args.n_colors, 24, kernel_size = 5, stride = 2, padding = 2)
        self.f_conv2 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)
        self.f_conv3 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)
        self.f_conv4 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)
        self.f_conv5 = nn.Conv2d(24, 8, kernel_size = 3, padding = 1)
        self.f_pixel_shuffle = nn.PixelShuffle(2)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, frame_1, frame_2):
        # Coarse flow
        coarse_in = tf.cat((frame_1, frame_2), dim = 1)
        coarse_in = self.relu(self.c_conv1(coarse_in))
        coarse_in = self.relu(self.c_conv2(coarse_in))
        coarse_in = self.relu(self.c_conv3(coarse_in))
        coarse_in = self.relu(self.c_conv4(coarse_in))
        coarse_in = self.tanh(self.c_conv5(coarse_in))
        coarse_out = self.c_pixel_shuffle(coarse_in)
        
        frame_2_compensated_coarse = warp(frame_2, coarse_out)
        
        # Fine flow
        fine_in = tf.cat((frame_1, frame_2, frame_2_compensated_coarse, coarse_out), dim = 1)
        fine_in = self.relu(self.f_conv1(fine_in))
        fine_in = self.relu(self.f_conv2(fine_in))
        fine_in = self.relu(self.f_conv3(fine_in))
        fine_in = self.relu(self.f_conv4(fine_in))
        fine_in = self.tanh(self.f_conv5(fine_in))
        fine_out = self.f_pixel_shuffle(fine_in)
        
        flow = coarse_out + fine_out
        frame_2_compensated = warp(frame_2, flow)
        return x
    
    def warp(img, flow):
        # TODO: add warping function
        # https://discuss.pytorch.org/t/solved-how-to-do-the-interpolating-of-optical-flow/5019
        return img
