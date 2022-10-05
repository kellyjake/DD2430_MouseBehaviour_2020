import cv2 as cv
import os , time , matplotlib , pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from ssm.plots import gradient_cmap
from ARHMM_utils import get_savestr , preprocess_data

def overlay_states(vid_filename, df, states, savename, labels=None, vert_padsize = 30, state_padsize=15, color_state=True):
    """
    df: dataframe from posedata (already subsampled and ouliers removed)
    """

    assert os.path.isfile(vid_filename)

    if not color_state:
        getcol = lambda state : (255,255,255)
    else:
        # This is the same colormap as in notebook! 
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
        cmap = gradient_cmap(colors)

        norm = matplotlib.colors.Normalize(vmin=0,vmax=max(states) + 1)

        # Makes it so that number in overlay has same color as state in eventplot in notebook
        getcol = lambda state : list(map(lambda a : a*255, cmap(norm(state))[:-1]))


 
    # Read video info
    cap = cv.VideoCapture(vid_filename)

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    vidWriter = cv.VideoWriter(savename, fourcc, fps, (width + state_padsize*2, height + vert_padsize + state_padsize), True)
    pbar = tqdm(total=len(states))        

    states = df.state.to_numpy()
    frames = df.frame.to_numpy() / 2

    for frame_no , state in zip(frames,states):

        if labels is None:
            label = None
        else:
            try:
                label = labels[int(state)]
            except IndexError:
                label = None
            
        frame_no = int(frame_no)

        ret = cap.set(cv.CAP_PROP_POS_FRAMES,frame_no)
        ok , frame = cap.read()

        pbar.update(1)
        if ok:
            texted_img = overlay_text(frame, state, frame_no, label, vert_padsize, state_padsize, fontScale=1, color=getcol(state))
            vidWriter.write(texted_img)

    vidWriter.release()
    



def overlay_text(img,state, frame_no, label=None, vert_padsize=30, state_padsize=15, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(255,255,255), show=False):
    """
    description
    """

    newimg = cv.copyMakeBorder(img, state_padsize, vert_padsize, state_padsize, state_padsize, cv.BORDER_CONSTANT, value=(0,0,0))
    
    height , width , _ = newimg.shape

    xmid = int(width/2)
    ymid = int(height/2)

    newimg[ymid-50:ymid+50,:state_padsize] = color
    newimg[ymid-50:ymid+50,width-state_padsize:] = color
    newimg[:state_padsize,xmid-50:xmid+50] = color

    img_t = cv.putText(newimg, f'Frame: {frame_no}', (0,height-5), fontFace, 0.3, (255,255,255), thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)
    img_t = cv.putText(img_t, 'State:', (170,height-20), fontFace, 0.4, (255,255,255), thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)            
    img_t = cv.putText(img_t, f'{int(state)}', (240,height-20), fontFace, 0.4, color, 1, lineType=cv.LINE_AA, bottomLeftOrigin=False)

    if label:
        img_t = cv.putText(img_t, 'Behaviour:', (170,height-5), fontFace, 0.4, (255,255,255), thickness, lineType=cv.LINE_AA, bottomLeftOrigin=False)            
        img_t = cv.putText(img_t, label, (240,height-5), fontFace, 0.4, color, 1, lineType=cv.LINE_AA, bottomLeftOrigin=False)


    img_t = cv.cvtColor(img_t, cv.COLOR_BGR2RGB)

    return img_t


def main():
    # 2050 vid
    data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_pose_data.csv'
    vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'
    #hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\mouse_2050_20201216-141715\mouse_2050_20201216-141715\hmm_model_11_states_2500_kappa.p'
    #hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\hmm_model_16_states_10000_kappa.p'
    hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\merged_hmm.p'

    # 2053 data
    #data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_pose_data.csv'
    #vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'

    # 2050 10000 kappa 16 states labels
    #labels = ['Walking','Still','Rearing','Stopping/body contracts','Reorient -> walk','Yaw left','Low sniffing','Yaw left','Exploratory','Yaw right','Initiate walking','Sideways scaling','Rearing','Yaw right','Sudden bodylength shift','Outlier']

    # 2050 2500 kappa 11 states labels
    #labels = ['Still','Walking','Initiating walking','Stopping/body contracts','?','?','Yaw left','Expand -> Constrict','Yaw right','Low sniffing','Outlier']

    # 2050 2500 kappa 11->9 states merge labels
    labels = ['Still','Walking','Walking','Body contraction','Undefined','Yaw left','Body elongation','Yaw right','Outlier/low sniffing']

    div = 10
    d = 5
    interval_start = 2000
    interval_end = 122000
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

    savename = get_savestr(newpath, f'state_vid','.mp4')
    
    overlay_states(vid_name, df,hmm_z, savename,labels)

if __name__ == '__main__':
    main()