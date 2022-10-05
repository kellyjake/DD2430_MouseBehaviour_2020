import numpy as np
import pandas as pd
import sys , os
sys.path.append('/home/titan/KI2020/Code/src')
from dlc_analysis import PoseExtractor
import matplotlib.pyplot as plt
from matplotlib import cm
import math

def roundup_n(x,n=10):
    return int(math.ceil(x / n)) * n

def rounddown_n(x,n=10):
    return int(math.floor(x / n)) * n


def triple_plot(x_lab,to_plot,savename,figsize=(15,15),increment=[10,10,10]):
    fig, ax = plt.subplots(nrows=len(to_plot),ncols=1,sharex=True,figsize=figsize)
    for i,idx in enumerate(to_plot):
        ax[i].scatter(df[x_lab],df[idx],s=1)
        ax[i].plot(df[x_lab],df[idx])
        ax[i].set_ylabel(idx)
        ax[i].set_yticks(np.arange(rounddown_n(np.nanmin(df[idx])), roundup_n(np.nanmax(df[idx])), increment[i]))

    ax[i].set_xlabel(x_lab)
    fig.savefig(os.path.join(savepath,savename))


#video_filename = '/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1_recording.avi'
config_path = '/home/titan/KI2020/DLC/KI2020_Project-Magnus-2020-08-28/config.yaml'
#savefile = '/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1_pose_data.npy'
savefile = '/home/titan/KI2020/ExperimentalResults/20200915/Mouse_195_10intensity_10on_15hz_30pulses_20ipt/20200915_behaviour2020_v_195_10intensity_10on_15hz_30pulses_20ipt_1/20200915_behaviour2020_v_195_10intensity_10on_15hz_30pulses_20ipt_1_pose_data.npy'
savepath = '/home/titan/KI2020/ExperimentalResults/20200915/Mouse_195_10intensity_10on_15hz_30pulses_20ipt/20200915_behaviour2020_v_195_10intensity_10on_15hz_30pulses_20ipt_1/'
#savepath = '/home/titan/KI2020/ExperimentalResults/20200911/Mouse_0411_10intensity_10on_15off_20/20200911_behaviour2020_v_0411_10intensity_10on_15off_20_1/'

pose = PoseExtractor()

#pose.create_baseline(video_filename,config_path,start_frame=33540,end_frame=33800)
#pose.extract_features()
df = pose.create_df(savefile)

cmap = cm.get_cmap('Spectral')

df.plot(x = 'cog_x', y='cog_y', kind='scatter', c='frame', cmap=cmap, s=1, alpha=.5)
plt.savefig(os.path.join(savepath,'cog_loc.png'))

df.plot(x = 'cog_x', y='cog_y', kind='line')
plt.legend(loc='upper left')
plt.savefig(os.path.join(savepath,'cog_line.png'))

gaze = ['gaze_vec_x','gaze_vec_y','gaze_vec_z']
cog = ['cog_x','cog_y','cog_z']
angles = ['body_angle','head_angle','head_body_angle']

triple_plot('frame',gaze,os.path.join(savepath,'gaze_vec.png'),increment=[10,10,10])
triple_plot('frame',cog,os.path.join(savepath,'cog_vec.png'),increment=[50,50,5])
triple_plot('frame',angles,os.path.join(savepath,'angles.png'),increment=[3,3,2])

df
