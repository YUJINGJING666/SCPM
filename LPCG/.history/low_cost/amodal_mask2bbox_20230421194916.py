import os
import sys
import numpy as np
from tqdm import tqdm
import yaml


kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_dir = kitti_cfg['data_dir']
dst_dir = kitti_cfg['KITTI_merge_data_dir'] # 经数据准备后输出的目标文件夹
root_3d_dir = kitti_cfg['KITTI3D_data_dir'] # 提供验证集的源文件夹

source_file = kitti_dir + '/data_file/split/train_raw.txt'

raw_info = np.loadtxt(source_file, dtype=str)
img_list = raw_info[:, 0] # 训练集对应的单目RGB图像数据的文件路径
print(img_list)