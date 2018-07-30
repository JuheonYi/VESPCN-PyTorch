# Below are imports just for testing
from option import args
import cv2
from data.vsrdata import VSRData

if __name__ == '__main__':
    vsr = VSRData(args)
    print(len(vsr.data_hr))
    print(len(vsr.data_lr))
    print(vsr.data_hr[1].shape)
    print(vsr.data_lr[1].shape)
    img_samples = []
    for i in range(args.n_sequence):
        cv2.imwrite('hr_{}.jpg'.format(i), vsr.data_hr[0][i, :])
        cv2.imwrite('lr_{}.jpg'.format(i), vsr.data_lr[0][i, :])
    print(len(vsr[0][0]))
    print(vsr[0][0][0].shape)
    print(len(vsr[0][1]))
    print(vsr[0][1][0].shape)
    print(vsr[0][2])