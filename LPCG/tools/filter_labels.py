import os
import numpy as np
from tqdm import tqdm
import yaml

kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_root_dir = kitti_cfg['root_dir']
kitti_merge_data_dir = kitti_cfg['KITTI_merge_data_dir']
mode = kitti_cfg['label_mode'] # LPCG的模式
target_cls = kitti_cfg['writelist'] # 伪标签保留的目标类别标签

if __name__ == '__main__':
    label_dir = ['{}/{}/label_2'.format(kitti_root_dir, i) for i in mode] # label_2: 目标类别['Car', 'Pedestrian', 'Cyclist'], 置信度>0.7, raw_label_2: 目标类别['Car', 'Pedestrian', 'Cyclist'], 无置信度阈值
    dst_filter_label_dir = ['{}/{}/filter_label_2_all_categories'.format(kitti_root_dir, i) for i in mode] # filter_label_2: 目标类别['Car'], filter_label_2_all_categories: 目标类别['Car', 'Pedestrian', 'Cyclist']
    for d in dst_filter_label_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    root_dir = kitti_merge_data_dir
    train_id_path = os.path.join(root_dir, 'split/train.txt')
    train_id = np.loadtxt(train_id_path, dtype=str)

    filter_train_id_path = [os.path.join(root_dir, 'split/train_{}_all_categories.txt'.format(i)) for i in mode] # train_{}: 目标类别['Car'], train_{}_all_categories: 目标类别['Car', 'Pedestrian', 'Cyclist']

    for l_d, dst_l_d, filter_id in zip(label_dir, dst_filter_label_dir, filter_train_id_path): # 遍历不同的模式
        id_list = []
        for id in tqdm(train_id): # 过滤当前模式下的伪标签，过滤空伪标签文件，过滤除目标类别外的其他类别的伪标签
            cur_label = np.loadtxt(os.path.join(l_d, id+'.txt'), dtype=str).reshape(-1, 15)
            if cur_label.shape[0] < 1:
                continue

            cur_label[cur_label[:, 4].astype(np.float32) < 0] = 0
            cur_label[cur_label[:, 6].astype(np.float32) < 0] = 0

            cur_label_ind = [i[0] in target_cls for i in cur_label]
            if np.sum(cur_label_ind) < 1:
                continue

            id_list.append(id)
            np.savetxt(os.path.join(dst_l_d, id+'.txt'), cur_label[cur_label_ind], fmt='%s') # 保留的类别是目标类别的伪标签
        np.savetxt(filter_id, id_list, fmt='%s')
