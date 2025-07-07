# from __future__ import print_function
# import os
# import sys
# sys.path.insert(0,'utils/')
# #sys.path.insert(0,'dataloader/')
# sys.path.insert(0,'models/')
# import cv2
# import pdb
# import argparse
# import numpy as np
# import skimage.io
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.utils.data
# from torch.autograd import Variable
# import time
# from tqdm import tqdm
# from utils.io import mkdir_p
# from utils.util_flow import write_flow, save_pfm, ConvertMiddleburyFloToKittiPng

# flo_filepath = '/Volumes/T7/tjc/VCN/weights/kitti-ft-trainval/kittirawdata/2011_10_03/2011_10_03_drive_0027_sync/flow/flo_file/0000003875.flo'
# png_filepath = flo_filepath.replace('png_file', 'flo_file').replace('png', 'flo')
# ConvertMiddleburyFloToKittiPng(flo_filepath, png_filepath)
# print("Finish!")

import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def load_flow_to_png(path):
    flow = load_flow_to_numpy(path)
    image = flow_to_image(flow)
    return image


def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


if __name__ == '__main__':
    image = load_flow_to_png('/Volumes/T7/tjc/VCN/weights/kitti-ft-trainval/kittirawdata/2011_10_03/2011_10_03_drive_0027_sync/flow/flo_file/0000003875.flo')
    plt.imshow(image)
    plt.show()
