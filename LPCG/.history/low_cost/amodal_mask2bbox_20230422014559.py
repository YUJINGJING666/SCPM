import os
import sys
import numpy as np
from tqdm import tqdm
import yaml
import pickle


def get_crop_coord(binary_mask):
    x_min = np.min(np.where(binary_mask == 1)[0])
    x_max = np.max(np.where(binary_mask == 1)[0])
    
    y_min = np.min(np.where(binary_mask == 1)[1])
    y_max = np.max(np.where(binary_mask == 1)[1])
    
    return x_min, y_min, x_max, y_max

kitti_cfg = yaml.load(open('../config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_dir = kitti_cfg['data_dir']
img_dir = os.path.join(kitti_cfg['KITTI_merge_data_dir'], 'image_2')

source_file = kitti_dir + '/data_file/split/train_raw.txt'

predres_path = "/data1/datasets/KITTI/dataset_for_SaVos/results/Kitti_car_track_40_0.005/log_pred_res" # 没有轨迹划分："dataset/data/Kitti/Kitti_car_track"，轨迹划分："dataset/data/Kitti/Kitti_car_track_with_track_split"

raw_info = np.loadtxt(source_file, dtype=str)
img_list = raw_info[:, 0] # 训练集对应的单目RGB图像数据的文件路径
print(img_list)
print(len(img_list))

count = 0
for idx, img_path in enumerate(img_list):
    img_file_path = os.path.join(img_dir, '{:0>6}.png'.format(idx))
    seg_mask_path = img_file_path.replace('image_2', 'seg_mask').replace('png', 'npz')
    seg_bbox_path = img_file_path.replace('image_2', 'seg_bbox').replace('png', 'txt')
    video_name = img_path.split('/')[-4]
    timestep = img_path.split('/')[-1].replace('.png', '')
    print('{:0>6}'.format(idx), "<->", video_name, "<->", '{:0>6}'.format(int(timestep)))

    pkl_path = os.path.join(predres_path, video_name, "%d_pred_res.pkl" % int(timestep))
    if not os.path.exists(pkl_path):
        print("pkl file for frame N of current scene does not exist, skipping...")
        continue

    single_img_pred = pickle.load(open(pkl_path, "rb"))
    print(single_img_pred.keys())

    all_vm = [val["vm"] for val in single_img_pred.values()]
    all_vm = sum(all_vm)
    all_vm = (all_vm > 0).astype(np.uint8)

    bbox = []
    masks = []
    for obj_id in single_img_pred.keys():
        pred_fm = single_img_pred[obj_id]["pred_fm"]
        pred_fm = (pred_fm > 0.5).astype(np.uint8)

        vm = single_img_pred[obj_id]["vm"]
        vm = (vm > 0.5).astype(np.uint8)

        other_visible_mask = all_vm - vm
        pred_fm = vm * (1-other_visible_mask) + \
                  pred_fm * other_visible_mask
        pred_fm = pred_fm.astype(np.uint8)
        
        print(obj_id, "::", pred_fm.shape, "::", pred_fm.sum())

        x_min, y_min, x_max, y_max = get_crop_coord(pred_fm)
        masks.append(pred_fm)
        bbox.append([x_min, y_min, x_max, y_max])

    masks = np.stack(masks, axis=0)
    bbox = np.stack(bbox, axis=0)
    scores = np.ones(bbox.shape[0])
    if bbox.shape[0] > 0:
        bbox2d = np.hstack([bbox.reshape(-1, 4), scores.reshape(-1, 1)])
        np.savetxt(seg_bbox_path, bbox2d, fmt='%f')
        np.savez(seg_mask_path, masks=masks[:, 0, ...])
    # print(masks.shape)
    # print(bbox.shape)
    # print(bbox2d.shape)
    count += 1
    if count == 2:
        break