import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

TIMESTEPS_PER_SEQ_TEST = {'2011_09_26_drive_0001_sync': 108, '2011_09_26_drive_0002_sync': 77, '2011_09_26_drive_0005_sync': 154,
                          '2011_09_26_drive_0009_sync': 447, '2011_09_26_drive_0011_sync': 233, '2011_09_26_drive_0013_sync': 144,
                          '2011_09_26_drive_0014_sync': 314, '2011_09_26_drive_0015_sync': 297, '2011_09_26_drive_0017_sync': 114,
                          '2011_09_26_drive_0018_sync': 270, '2011_09_26_drive_0019_sync': 481, '2011_09_26_drive_0020_sync': 86,
                          '2011_09_26_drive_0022_sync': 800, '2011_09_26_drive_0023_sync': 474, '2011_09_26_drive_0027_sync': 188,
                          '2011_09_26_drive_0028_sync': 430, '2011_09_26_drive_0029_sync': 430, '2011_09_26_drive_0032_sync': 390,
                          '2011_09_26_drive_0035_sync': 131, '2011_09_26_drive_0036_sync': 803, '2011_09_26_drive_0039_sync': 395,
                          '2011_09_26_drive_0046_sync': 125, '2011_09_26_drive_0048_sync': 22, '2011_09_26_drive_0051_sync': 438,
                          '2011_09_26_drive_0052_sync': 78, '2011_09_26_drive_0056_sync': 294, '2011_09_26_drive_0057_sync': 361,
                          '2011_09_26_drive_0059_sync': 373, '2011_09_26_drive_0060_sync': 78, '2011_09_26_drive_0061_sync': 703,
                          '2011_09_26_drive_0064_sync': 570, '2011_09_26_drive_0070_sync': 420, '2011_09_26_drive_0079_sync': 100,
                          '2011_09_26_drive_0084_sync': 383, '2011_09_26_drive_0086_sync': 706, '2011_09_26_drive_0087_sync': 729,
                          '2011_09_26_drive_0091_sync': 340, '2011_09_26_drive_0093_sync': 433, '2011_09_26_drive_0095_sync': 268,
                          '2011_09_26_drive_0096_sync': 475, '2011_09_26_drive_0101_sync': 936, '2011_09_26_drive_0104_sync': 312,
                          '2011_09_26_drive_0106_sync': 227, '2011_09_26_drive_0113_sync': 87, '2011_09_26_drive_0117_sync': 660,
                    
                          '2011_09_28_drive_0001_sync': 106, '2011_09_28_drive_0002_sync': 376,
                        
                          '2011_09_29_drive_0004_sync': 339, '2011_09_29_drive_0026_sync': 158, '2011_09_29_drive_0071_sync': 1059,
                        
                          '2011_09_30_drive_0016_sync': 279, '2011_09_30_drive_0018_sync': 2762, '2011_09_30_drive_0020_sync': 1104,
                          '2011_09_30_drive_0027_sync': 1106, '2011_09_30_drive_0028_sync': 5177, '2011_09_30_drive_0033_sync': 1594,
                          '2011_09_30_drive_0034_sync': 1224,
              
                          '2011_10_03_drive_0027_sync': 4544, '2011_10_03_drive_0034_sync': 4663, '2011_10_03_drive_0042_sync': 1170,
                          '2011_10_03_drive_0047_sync': 837} # KITTI raw data每个视频序列包含的帧数（图像文件个数）


def is_image_file(filename, suffix=None):
    if suffix is not None:
        IMG_EXTENSIONS = [suffix]
    else:
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npz'
        ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, suffix=None, max=None):
    """Access the video sequence folder and return a list of image file paths
    arranged in time sequence under the current folder.

    Args:
        dir (str): Video sequence folder path.
        suffix (str, optional): File extension. Defaults to None.
        max (int, optional): Maximum number of files. Defaults to None.

    Returns:
        _type_: _description_
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, suffix):
                path = os.path.join(root, fname)
                images.append(path)
    if max is not None:
        return images[:max]
    else:
        return images

def dataloader(filepath, reverse):

    image_root = filepath # dataset/kitti_scene/raw_data/
    timestamps = TIMESTEPS_PER_SEQ_TEST

    image_10_path = [] # 用于存储image_10的文件路径
    image_11_path = [] # 用于存储image_11的文件路径
    for subdir in timestamps.keys():
        image_list = sorted(make_dataset(os.path.join(image_root, "_".join(subdir.split("_")[:3]), subdir, 'image_02/data'), suffix='.png'))
        image_list = ['/'.join(el.split('/')[-5:]) for el in image_list] # 用于存储KITTI raw data的文件路径，元素格式：‘[年_月_日]/[年_月_日_drive_视频索引_sync]/image_02/data/[图像索引].png’
        if reverse:
            print("========== Reverse timing mode ==========")
            image_10_path += image_list[1:]
            image_11_path += image_list[:-1]
        else:
            print("========== Sequential timing mode ==========")
            image_10_path += image_list[:-1]
            image_11_path += image_list[1:]

    l0_train = [image_root+img for img in image_10_path]
    l1_train = [image_root+img for img in image_11_path]
    flow_train = [image_root+img.replace('image_02','flow_occ') for img in image_10_path]

    return sorted(l0_train), sorted(l1_train), sorted(flow_train)
