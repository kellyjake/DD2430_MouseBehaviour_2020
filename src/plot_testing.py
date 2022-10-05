from dlc_pipeline import triple_plot , extract_events , partition_data , create_plots
from dlc_analysis import PoseExtractor
import matplotlib.pyplot as plt 
import os
import numpy as np
import pandas as pd

src = os.path.normpath(r'C:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\20201202_behaviour2020_v_6287_distractionduring_task_vol2_1')
savefile_binary = os.path.join(src,'20201202_behaviour2020_v_6287_distractionduring_task_vol2_1_pose_data.npy')
event_filename = os.path.join(src,'20201202_behaviour2020_v_6287_distractionduring_task_vol2_1_data.csv')

#pose = PoseExtractor(None,None)
#df = pose.create_df(savefile_binary)
events = extract_events(event_filename)

create_plots(savefile_binary,events,100000)

#tmp_df , tmp_events = partition_data(df,events,100000,0)

#triple_plot(tmp_df,['frame'],['head_angle','head_body_angle'],'test.png',tmp_events)

