import os
import sys
import numpy as np
from tqdm import tqdm
import yaml
import pickle


kitti_cfg = yaml.load(open('../config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_dir = kitti_cfg['data_dir']
dst_dir = kitti_cfg['KITTI_merge_data_dir'] # 经数据准备后输出的目标文件夹
root_3d_dir = kitti_cfg['KITTI3D_data_dir'] # 提供验证集的源文件夹

source_file = kitti_dir + '/data_file/split/train_raw.txt'

predres_path = "/data1/datasets/KITTI/dataset_for_SaVos/results/Kitti_car_track_40_0.005/log_pred_res" # 没有轨迹划分："dataset/data/Kitti/Kitti_car_track"，轨迹划分："dataset/data/Kitti/Kitti_car_track_with_track_split"

raw_info = np.loadtxt(source_file, dtype=str)
img_list = raw_info[:, 0] # 训练集对应的单目RGB图像数据的文件路径
print(img_list)
print(len(img_list))

for idx, img_path in enumerate(img_list):
    video_name = img_path.split('/')[-4]
    timestep = img_path.split('/')[-1].replace('.png', '')
    print('{:0>6}'.format(idx), "<->", video_name, "<->", '{:0>6}'.format(int(timestep)))
    single_img_pred = pickle.load(open(os.path.join(predres_path, video_name, "%d_pred_res.pkl" % timestep), "rb"))
    print(single_img_pred.keys())
    break