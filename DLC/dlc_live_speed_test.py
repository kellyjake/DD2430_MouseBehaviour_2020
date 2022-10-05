from dlclive import DLCLive, Processor
import cv2 , time , sys
sys.path.append('/home/titan/KI2020/Code/src')
from pose import get_output_variables
import numpy as np

modelfile = "/home/titan/KI2020/DLC/KI2020_Project-Magnus-2020-08-28/exported-models/DLC_KI2020_Project_resnet_50_iteration-0_shuffle-1/"
vidfile = '/home/titan/KI2020/ExperimentalResults/20200827/Mouse_29510on15off40hz5intensity/20200827_behaviour2020_v_29510on15off40hz5intensity_1/20200827_behaviour2020_v_29510on15off40hz5intensity_1_recording.avi'

dlc_proc = Processor()
dlc_live = DLCLive(modelfile)
dlc_live.init_inference()

cap = cv2.VideoCapture(vidfile)

succ , img = cap.read()

x0 = np.array((4.74, 3.16, 1.58))

n = 1000
imgs = []

for _ in range(n):
    succ , img = cap.read()
    imgs.append(img)

t_start = time.time()

for im in imgs:
    pose = dlc_live.get_pose(im)
    vals = get_output_variables(pose[:,:2],x0)

print((time.time() - t_start)/n)