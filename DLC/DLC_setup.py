#!~/anaconda3/envs/DLC-GPU/bin/ipython

import deeplabcut as dlc
import tensorflow as tf
import os

parent_folder = '/home/titan/KI2020/ExperimentalResults/'
all_videos = []

def check_for_videos(folder):
    _ , folders , files = next(os.walk(folder))

    if not(files):
        if folders:
            for subfolder in folders:
                check_for_videos(os.path.join(folder,subfolder))
    else:
        for f in files:
            f_split = f.split('.')

            if f_split[-1] == 'avi':
                all_videos.append(os.path.join(folder,f))

check_for_videos(parent_folder)


#config_path = dlc.create_new_project('KI2020_Project', 'Magnus', all_videos, copy_videos=False, working_directory='/home/titan/KI2020/DLC/')
config_path = '/home/titan/KI2020/DLC/KI2020_Project-Magnus-2020-08-28/config.yaml'

dlc.extract_frames(config_path, mode='automatic', algo='kmeans', crop=False)

dlc.label_frames(config_path)

dlc.check_labels(config_path)

dlc.create_training_dataset(config_path)

dlc.train_network(config_path, gputouse=0, maxiters=300000)