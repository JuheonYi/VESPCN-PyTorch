# VESPCN-PyTorch
PyTorch implementation of ESPCN [1]/VESPCN [2].

## **How to run the code**
1. Add your own template in template.py, indicating parameters related to running the code 
     (especially, specify the task (Image/MC/Video) and set training/test dataset directories specific to your filesystem)
2. Add your model in ./model/ directory (filename should be in lower cases)
3. Type "python3 main.py --template $(your template) --model $(model you want to train)" for training
4. If you want to add additional options for test benchmark datasets, modify ./data/__init__.py.
5. For additional details, refer to [3] (We have borrowed most of the implementation details from there).

## **TODO list**
- [x] Implement the SISR ESPCN network
- [x] Making dataloader for video SR
- [x] Complete the motion compensation network
- [x] Joining the ESPCN to motion compensation network

## **References**
[1] W. Shi et al, “Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network,” IEEE CVPR 2016.

[2] J. Caballero et al, “Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation,” IEEE CVPR 2017.

[3] https://github.com/thstkdgus35/EDSR-PyTorch (borrowed the overall code structure)
