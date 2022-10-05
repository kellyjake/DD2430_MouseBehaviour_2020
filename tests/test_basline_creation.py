import numpy as np
import pandas as pd
import sys 
sys.path.append('/home/titan/KI2020/Code/src')
from dlc_analysis import PoseExtractor
import matplotlib.pyplot as plt
from matplotlib import cm
import math


video_filename = '/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1_recording.avi'
config_path = '/home/titan/KI2020/DLC/KI2020_Project-Magnus-2020-08-28/config.yaml'
bl_path = '/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1_recording_baseline.csv'
pose = PoseExtractor()

pose.create_baseline(video_filename,config_path,start_frame=33540,end_frame=33800)

savefile_binary = pose.extract_features()

bl = np.fromfile(bl_path)

a0 = bl[:int(len(bl)/3 - 1)]
b0 = bl[int(len(bl)/3):2*int(len(bl)/3 - 1)]
c0 = bl[2*int(len(bl)/3):]

#savefile = '/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1_pose_data.npy'

bl , [a0,b0,c0] = pose.get_baseline(return_all=True)

fig, ax = plt.subplots(nrows=1,ncols=3)

ax[0].hist(a0)
ax[1].hist(b0)
ax[2].hist(c0)
fig.savefig('hist.png')
