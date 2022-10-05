from ARHMM_utils import preprocess_data , produce_ARHMM_data
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

csv_frans = r'c:\Users\magnu\OneDrive\Dokument\KI\KI2020\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_pose_data.csv'
csv_2050 = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_pose_data.csv'
csv_6287 = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\20201202_behaviour2020_v_6287_distractionduring_task_vol2_1\20201202_behaviour2020_v_6287_distractionduring_task_vol2_1_data.csv'

data_csv = csv_2050

interval_start = 0
interval_end = 10000

div = 10

ARHMM_kwargs={  'kappa':1,
                'use_best_K':True, 
                'K':None,
                'epochs':5,
                'start_K':1,
                'end_K':5}

make_vid = False
smooth = True
box_size = 0.6
box_pixel = 400
T = 2e-3
d = 5
seed = 1337
threshold = 0.001
early_stopping = False

kappa = ARHMM_kwargs['kappa']


data_frame2 = pd.read_csv(data_csv)

if interval_end in ['end',-1]:
    interval_end = len(data_frame2) - 1

interval = range(interval_start,interval_end,div)

# Preprocess data
data_to_use , speed , head_body_angle , body_len = preprocess_data(data_frame2, interval, div, smooth, box_size, box_pixel, d, T)

#seeds = [i for i in range(2)]
seeds = range(2)

pbar = tqdm(total=len(seeds))

val_lls_ll_list = []
best_K_ll_list = []
best_K_AIC_list = []
AIC_list = []

for seed in seeds:
    #¤¤ Produce ARHMM data! ¤¤¤
    hmm , best_train_lls , best_val_lls , best_K_AIC , best_K_ll , val_lls , train_lls , AIC = produce_ARHMM_data(data_to_use, ARHMM_kwargs, seed, early_stopping=early_stopping,threshold=threshold,use_AIC=True,return_all=True)
    val_lls_ll_list.append(val_lls)
    best_K_ll_list.append(best_K_ll)
    best_K_AIC_list.append(best_K_AIC)
    AIC_list.append(AIC)

    pbar.update(1)

fig , [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(10,5))

for val_ll in val_lls_ll_list:
    ax1.plot(range(ARHMM_kwargs['start_K'],ARHMM_kwargs['end_K']),val_ll)
    
ax1.set_xlabel('K')
ax1.set_ylabel('Loglikelihood')

for aic in AIC_list:
    ax2.plot(range(ARHMM_kwargs['start_K'],ARHMM_kwargs['end_K']),aic)

ax2.set_xlabel('K')
ax2.set_ylabel('AIC')

ax3.scatter(seeds,best_K_ll_list,labels='LL')
ax3.scatter(seeds,best_K_AIC_list,labels='AIC')
ax3.set_xlabel('Seed')
ax3.set_ylabel('Best K')
ax3.legend()

fig.show()



hmm_z = hmm.most_likely_states(data_to_use,)