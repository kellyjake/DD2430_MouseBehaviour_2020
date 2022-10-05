import pandas as pd
import numpy as np
from ARHMM_plots import preprocess_data , produce_ARHMM_data , rgb_to_hex , plot_state_by_pos

if __name__ == '__main__':

    pose_data = r'c:\Users\magnu\OneDrive\Dokument\KI\KI2020\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_pose_data.csv'
    dlc_data = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000.csv'


    interval_start = 0
    interval_end = 'end'
    div = 1

    ARHMM_kwargs={  'kappa':5,
                    'use_best_K':True, 
                    'K':None,
                    'epochs':20,
                    'start_K':1,
                    'end_K':5}

    make_vid = True
    smooth = True
    box_size = 0.6
    box_pixel = 400
    T = 2e-3
    d = 5
    seed = 1337
    early_stopping = True


    pose_df = pd.read_csv(pose_data)

    if interval_end in ['end',-1]:
        interval_end = len(pose_df) - 1

    interval = range(interval_start,interval_end)

    # Preprocess data
    data_to_use , df , speed , head_body_angle , body_len , outliers = preprocess_data(pose_df, interval, div, smooth, box_size, box_pixel, d, T)

    #¤¤ Produce ARHMM data! ¤¤¤
    hmm , best_train_lls , best_val_lls , best_K_AIC , best_K_ll , val_lls , train_lls , AIC = produce_ARHMM_data(data_to_use, ARHMM_kwargs, seed, early_stopping=early_stopping)

    hmm_z = hmm.most_likely_states(data_to_use,)

    df['state'] = hmm_z

    plot_state_by_pos(df, dlc_data, 'test.html')
    