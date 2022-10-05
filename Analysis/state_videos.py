
from ssm.plots import gradient_cmap
import numpy as np
import seaborn as sns
import pandas as pd
import csv , os ,  tqdm , time , ssm , json , sys , pickle
from overlay_states import overlay_states
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize
from ARHMM_utils import rgb_to_hex , preprocess_data , get_savestr
import cv2 as cv

def make_state_videos(df,vid_filename,savename,vert_padsize=50,n_occ=-1,shuffle_occ=False):

    
    # Read video info
    cap = cv.VideoCapture(vid_filename)

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
    fps = int(cap.get(cv.CAP_PROP_FPS) / 4)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    crop_x = 150
    crop_y = 150
    frame_range = 100

    newsize_x = 2*(crop_x)
    newsize_y = 2*(crop_y) + vert_padsize

    df['frame'] /= 2

    color_names = [
            "windows blue",
            "red",
            "amber",
            "faded green",
            "dusty purple",
            "orange",
            "dark navy",
            "light urple",
            "rosa",
            "cinnamon",
            "bruise",
            "dark sage"
        ]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors,nsteps=len(np.unique(df['state'])))
    vals = np.unique(df['state'])
    m_colors = {f'{val/np.max(vals)}':rgb_to_hex(cmap(val/np.max(vals))[:-1]) for val in vals}

    df['state_consec'] = (df.state.diff(1) != 0).astype('int')

    df['state_appearence'] = df.groupby('state')['state_consec'].cumsum()

    #df['cog_x_shift'] = df.cog_x.shift(frame_range)
    #df['cog_y_shift'] = df.cog_y.shift(frame_range)
    #df['cog_x_post_shift'] = df.cog_x.shift(frame_range)
    #df['cog_y_post_shift'] = df.cog_x.shift(frame_range)
    
    df_start_idx = df.index[0]

    df['cog_x'] = df['cog_x'].astype('int')
    df['cog_y'] = df['cog_y'].astype('int')

    pbar1 = tqdm.tqdm(total=len(vals))
    
    for i,k in enumerate(vals):
        presave , ext = os.path.splitext(savename)
        presave += f'_state_{k}'
        tmp_savename = presave + ext

        vidWriter = cv.VideoWriter(tmp_savename, fourcc, fps, (newsize_x, newsize_y), True)

        newdf = df[df['state'] == k]
        occurences = np.unique(newdf['state_appearence'])

        if shuffle_occ:
            np.random.shuffle(occurences)

        max_occ = max(occurences) - 1

        if len(occurences) == 0:
            print(f"No occurence of state {k}!")
            continue

        pbar2 = tqdm.tqdm(total=len(occurences))

        for occurence in occurences[:min(n_occ,max_occ)]:
            tmpdf = newdf[newdf['state_appearence'] == occurence]
            
            idx_start = tmpdf.index[0]
            idx_end = tmpdf.index[-1]

            predf = df.loc[max(0,idx_start - frame_range):idx_start-1]

            m = len(predf)
            n = len(tmpdf)
            
            start_frame = max(df_start_idx,idx_start - frame_range)

            # Draw black frames before showing            
            for i in range(10):
                empty_frame = np.zeros((newsize_y,newsize_x,3),dtype='uint8')
                empty_frame = cv.putText(empty_frame, f'State {k}, Occ. {occurence}', (int(newsize_x/2) - 50,int(newsize_y/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv.LINE_AA, bottomLeftOrigin=False)
                vidWriter.write(empty_frame)    

            
            # Draw frames before state shift happens (predf) and during state (tmpdf)
            for length , this_df , docolor in zip([m,n],[predf,tmpdf],[False,True]):
                for row_no in range(length):
                    try:
                        frame_no = int(this_df.iloc[row_no].frame)
                        state = int(this_df.iloc[row_no].state)
                        pre_x = this_df.iloc[row_no].cog_x
                        pre_y = this_df.iloc[row_no].cog_y

                        ret = cap.set(cv.CAP_PROP_POS_FRAMES,frame_no)
                        ok , frame = cap.read()

                        if ok:
                            pre_frame = overlay_events(frame,frame_no,pre_x,pre_y,state,vert_padsize,crop_x,crop_y,width,height,docolor)
                            vidWriter.write(pre_frame)
                    except (KeyError , IndexError):
                        print(f"Pre: Skipping row no {row_no}")
                        pass

            pbar2.update(1)

        vidWriter.release()
        pbar1.update(1)

def overlay_events(frame,frame_no,x,y,state,vert_padsize,crop_x,crop_y,width,height,dopaint):

    textcolor = (0,255,0) if dopaint else (255,255,255)
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    thickness = 1
    #newframe = np.zeros((2*crop_x,2*crop_y,3))
    
    frame = cv.copyMakeBorder(frame, crop_x,crop_y,crop_x,crop_y,cv.BORDER_CONSTANT, value=(0,0,0))
    
    x += crop_x
    y += crop_y

    ycrop_ver_min = int(y - crop_y)
    ycrop_ver_max = int(y + crop_y)
    xcrop_ver_min = int(x - crop_x)
    xcrop_ver_max = int(x + crop_x)
    
    new_height = ycrop_ver_max - ycrop_ver_min
    new_width = xcrop_ver_max - xcrop_ver_max

    if dopaint:
        frame = cv.circle(frame,(int(x),int(y)),5,textcolor,cv.FILLED)

    frame = frame[ycrop_ver_min : ycrop_ver_max, xcrop_ver_min : xcrop_ver_max,:]
    frame = cv.copyMakeBorder(frame, 0,vert_padsize,0,0,cv.BORDER_CONSTANT, value=(0,0,0))
    frame = cv.putText(frame, f'Frame: {frame_no}', (0,new_height + vert_padsize - 5), fontFace, 0.3, (255,255,255), thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)
    frame = cv.putText(frame, f'State:', (100,new_height + vert_padsize - 5), fontFace, 0.3, (255,255,255), thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)            
    frame = cv.putText(frame, f'{state}', (135,new_height + vert_padsize - 5), fontFace, 0.3, textcolor, thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)            
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    return frame

if __name__ == '__main__':
    
    # 2050 vid
    data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_pose_data.csv'
    vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'
    #hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\mouse_2050_20201216-141715\mouse_2050_20201216-141715\hmm_model_11_states_2500_kappa.p'
    #hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\hmm_model_16_states_10000_kappa.p'
    
    hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\merged_hmm.p'
    # 2053 vid
    #data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_pose_data.csv'
    #vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'

    # 1337 vid
    #data_csv = r'c:\Users\magnu\OneDrive\Dokument\KI\KI2020\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_pose_data.csv'
    #vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'

    div = 10
    d = 5
    interval_start = 0
    interval_end = 220000
    interval = range(interval_start,interval_end)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    p = os.path.normpath(data_csv)
    newpath = os.sep.join([os.sep.join(p.split(os.sep)[:-1]), 'results',timestr])
    Path(newpath).mkdir(parents=True, exist_ok=True)

    data_frame = pd.read_csv(data_csv)

    with open(hmm_file,'rb') as f:
        hmm = pickle.load(f)

    data_to_use , df , speed , head_body_angle , body_len , outliers = preprocess_data(data_frame, interval, div=div, d=d)

    hmm_z = hmm.most_likely_states(data_to_use)

    df['state'] = hmm_z

    make_state_videos(df, vid_name, get_savestr(newpath, f'state_visits','.mp4'),n_occ=15,shuffle_occ=True)
