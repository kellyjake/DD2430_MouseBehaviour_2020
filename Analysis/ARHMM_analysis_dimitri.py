from ARHMM_plots import make_plots
import tkinter as tk
from tkinter import filedialog
import os
"""
HOW TO USE:

Just change the name of the data_csv and vid_name to the file you want to get 
results from and tweak the parameters accordingly.

You will have to specify which part of the video we are interested in (interval_start and end)
and the rate of downsampling (if any).

There are some more parameters available but none that you need to worry about.

/Magnus
"""

"""
# Path to csv file with pose_data.csv
data_csv = r'/home/titan/KI2020/ExperimentalResults/20201126/Mouse_2507_15on_10off_40hz_1sec/20201126_behaviour2020_v_2507_15on_10off_40hz_1sec_1/20201126_behaviour2020_v_2507_15on_10off_40hz_1sec_1_data.csv'

# Path to video
vid_name = r'/home/titan/KI2020/ExperimentalResults/20201126/Mouse_2507_15on_10off_40hz_1sec/20201126_behaviour2020_v_2507_15on_10off_40hz_1sec_1/20201126_behaviour2020_v_2507_15on_10off_40hz_1sec_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'

# Path to DLC raw output
dlc_csv = r'c:\Users\magnu\OneDrive\Dokument\KI\KI2020\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.csv'
"""

# Frame number to start
interval_start = 31135

# Frame number to end
interval_end = 37755

# Downsampling rate (div=1 gives 500 FPS -> div=10 gives 50 FPS etc.)
downsampling_rate = 10

# Produce output video?
create_video=True

# Set random seed -> same seed on same data will produce the same output every time you run it
# Good for reproducing data.
# (Set it to some integer value) NOT STRING
seed=1337


root = tk.Tk()
root.withdraw()

# Enter data
data_csv = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select EVENT DATA file (_data.csv)", filetypes = [("csv file","*.csv")])

cwd = os.sep.join(os.path.splitext(data_csv)[0][:-1])

video_filename = filedialog.askopenfilename(initialdir = cwd, title = "Select VIDEO file", filetypes = [("Video file",["*.avi","*.mp4"])])

dlc_csv = filedialog.askopenfilename(initialdir = cwd, title = "Select RAW DLC file (_pose_data.csv)", filetypes = [("csv file","*.csv")])

root.destroy()

# Call function to make plots and goodies
make_plots( data_csv=data_csv,
            vid_name=video_filename,
            raw_dlc_data=dlc_csv,
            interval_start=interval_start,
            interval_end=interval_end,
            div=downsampling_rate,
            seed=seed,
            make_vid=create_video)