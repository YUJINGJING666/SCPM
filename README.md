CODE DOCUMENT
1.STC(Pseudo labels) 

Config 
root_dir: '/mnt/disk1/tjc/code/LPCG'

data_dir: '/mnt/disk1/tjc/code/LPCG/data/kitti'
raw_data_dir: '/home/tjc/LPCG/data/kitti/raw_data' # KITTI raw data
KITTI3D_data_dir: # KITTI3Dpublic training set
'/home/tjc/LPCG/data/kitti/KITTI3D/training'
KITTI_merge_data_dir: # Expanded dataset
'/home/tjc/LPCG/data/kitti/kitti_merge/training'

soft link

ln -s /data1/KITTI/detection3d/kitti_data /home/tjc/LPCG/data/kitti/raw_data
ln -s /data1/KITTI/detection3d/detection3d/training/calib /home/tjc/LPCG/data/kitti/KITTI3D/training
ln -s /data1/KITTI/detection3d/detection3d/training/velodyne /home/tjc/LPCG/data/kitti/KITTI3D/training
ln -s /data1/KITTI/detection3d/detection3d/training/label_2 /home/tjc/LPCG/data/kitti/KITTI3D/training
ln -s /data1/KITTI/detection3d/detection3d/training/image_2 /home/tjc/LPCG/data/kitti/KITTI3D/training
ln -s /data1/KITTI/detection3d/detection3d/testing/calib /home/tjc/LPCG/data/kitti/KITTI3D/testing
ln -s /data1/KITTI/detection3d/detection3d/testing/velodyne /home/tjc/LPCG/data/kitti/KITTI3D/testing
ln -s /data1/KITTI/detection3d/detection3d/testing/image_2 /home/tjc/LPCG/data/kitti/KITTI3D/testing

1. Prepare KITTI raw data set for training
python data/kitti/prepare_kitti_raw_datafile.py (#optionally)
python data/kitti/link_kitti_raw.py
2. Clone OpenPCDet in the high_acc folder and follow the installation instructions.
git clone https://github.com/open-mmlab/OpenPCDet.git
3. Install related libraries
pip install spconv-cu111
4. Install the pcdet library and its dependencies.
python setup.py develop
If there is a mismatch in the library version, install it separately and then execute this command.
5. Generate pseudo labels on the original KITTI data (excluding validation sequences)
cp high_acc/infer_kitti.py high_acc/OpenPCDet/tools/
cd high_acc/OpenPCDet/tools
CUDA_VISIBLE_DEVICES=0 python infer_kitti.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../pv_rcnn_8369.pth --data_path /home/cxh/tjc/LPCG/data/kitti/kitti_merge/training/velodyne
cd ../../..
6. Filter out empty pseudo labels used for training, then link verification labels.
python tools/filter_labels.py
python tools/link_kitti_val_labels.py
7.	Training monocular 3D detection models using pseudo labels

2.GDP
KITTI_merge_data_dir: # Expanded dataset 
'/home/tjc/GUPNet/data/KITTI/training'

soft link
ln -s /home/tjc/LPCG/data/kitti/kitti_merge/training/calib /home/tjc/GUPNet/data/KITTI/training
ln -s /home/tjc/LPCG/data/kitti/kitti_merge/training/velodyne /home/tjc/GUPNet/data/KITTI/training/depth
ln -s /home/tjc/LPCG/high_acc/filter_label_2 /home/tjc/GUPNet/data/KITTI/training/label_2
ln -s /home/tjc/LPCG/data/kitti/kitti_merge/training/image_2 /home/tjc/GUPNet/data/KITTI/training
ln -s /data1/KITTI/detection3d/detection3d/testing/calib /home/tjc/GUPNet/data/KITTI/testing
ln -s /data1/KITTI/detection3d/detection3d/testing/velodyne /home/tjc/GUPNet/data/KITTI/testing/depth
ln -s /data1/KITTI/detection3d/detection3d/testing/image_2 /home/tjc/GUPNet/data/KITTI/testing

Execute the following in the code folder
1. Train models
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_val.py
2. Evaluation model
After training, the model will directly feed back the detection files for evaluation (if so, you can skip this step). However, if you want to evaluate a given checkpoint, you need to modify the ‘resume_model’ of ‘tester’ in the code/experiments/config.yaml, and then run:
python tools/train_val.py -e
After that, please use the Kitti evaluation devkit (details can be found in FrustumPointNet) to evaluate:
tools/kitti_eval/evaluate_object_3d_offline_ap40.cpp -o tools/kitti_eval/evaluate_object_3d_offline_ap40.exe
An executable file has been generated. Simply run the code below.
tools/kitti_eval/evaluate_object_3d_offline_ap40.exe ../data/KITTI/training/label_2 ./outputs
3. Test model
Modify the train set to the trainval set (you can modify it in code/libs/helpers/dataloader_helper.py), and then modify the input of the evaluation function to the test set.
（code/tools/train_val.py）。
The model parameter output path and evaluation result file output path are also modified accordingly.
（code/lib/helpers/trainer_helper.py
code/lib/helpers/tester_helper.py），The loading path for ‘resume_model’ under ‘tester’ in config/experiments/config.yaml has also been changed accordingly.
Compress the output file into a zip file (please note that this zip file does not contain any root directories):
cd outputs/data
zip -r submission.zip .


