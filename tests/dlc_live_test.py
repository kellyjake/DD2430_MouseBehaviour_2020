from dlclive import DLCLive , Processor
import deeplabcut
import cv2 as cv
from pose2d import get_output_variables_2d
import numpy as np

config_path = '/home/titan/KI2020/DLC/KI2020_Training-Valter-2020-09-19/exported-models/DLC_KI2020_Training_resnet_50_iteration-0_shuffle-1/'
vid = '/home/titan/KI2020/ExperimentalResults/20201030/Mouse_6287_10mw_15on_40hz_nococ/20201030_behaviour2020_v_6287_10mw_15on_40hz_nococ_1/20201030_behaviour2020_v_6287_10mw_15on_40hz_nococ_1_recording.avi'

vid_reader = cv.VideoCapture(vid)

success , frame = vid_reader.read()

dlc_proc = Processor()
dlc_live = DLCLive(config_path, processor=dlc_proc)        
dlc_live.init_inference()

success , frame = vid_reader.read()
vals = dlc_live.get_pose(frame)
#[snout_x , snout_y , snout_p] , [l_ear_x , l_ear_y , l_ear_p] , [r_ear_x , r_ear_y , r_ear_p] , [tail_x , tail_y , tail_p] = dlc_live.get_pose(frame)

print(vals)

in_arr = np.array([ [snout_x , snout_y], 
                    [l_ear_x,l_ear_y], 
                    [r_ear_x, r_ear_y], 
                    [tail_x, tail_y]])

var_list = get_output_variables_2d(in_arr)

for p in [snout_p, l_ear_p, r_ear_p, tail_p]:
    print(p)
    
for val in var_list:
    print(val)
