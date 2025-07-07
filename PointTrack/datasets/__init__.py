from datasets.KittiMOTSDataset import *


def get_dataset(name, dataset_opts):
    if name == "mots_test":
        return MOTSTest(**dataset_opts)
    elif name == "mots_cars_val":
        return MOTSCarsVal(**dataset_opts)
    elif name == "mots_track_val_env_offset":
        return MOTSTrackCarsValOffset(**dataset_opts)
    elif name == "mots_track_cars_train":
        return MOTSTrackCarsTrain(**dataset_opts)
    elif name == "mots_cars":
        return MOTSCars(**dataset_opts)
    elif name == "KITTI_raw_data_InsSeg_test":
        return KITTI_raw_data_InsSeg_test(**dataset_opts)
    elif name == "KITTI_raw_data_Tracking_test":
        return KITTI_raw_data_Tracking_test(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))