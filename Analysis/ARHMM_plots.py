#!~/anaconda3/envs/DLC-GPU/bin/python
# coding: utf-8

from ssm.plots import gradient_cmap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv , os ,  tqdm , time , ssm , json
from overlay_states import overlay_states
import matplotlib.patches as mpatches
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
from plotly import tools
from matplotlib import cm
from ARHMM_utils import get_savestr , rgb_to_hex, preprocess_data , produce_ARHMM_data
import sys
from matplotlib.colors import Normalize
import pickle
from hmm_3d_trajectories import make_hmm_3d_trajectories , make_separate_3d_trajectories
from state_videos import make_state_videos

def make_cross_likelihood_matrix(hmm,df,savename):
    """
    Create matrix explaining how distinctly states capture behavioural motifs
    """

    df['state_consec'] = (df.state.diff(1) != 0).astype('int')

    df['state_appearence'] = df.groupby('state')['state_consec'].cumsum()

    values = np.unique(df['state'])

    max_K = hmm.K
        
    CL = np.zeros((max_K,max_K))

    # For each state k=1,...,K
    for k in range(max_K):
        
        # Find how many consecutive sequences that have been classified as k 
        N_k = df.loc[df['state'] == k]['state_appearence'].max()
        
        # If some stated not visited, leave column as 0
        if N_k is np.nan:
            continue
        
        # For each sequence, find cross-likelihood
        for i in range(1,N_k + 1):
            tmp_data = df.loc[(df['state'] == k) & (df['state_appearence'] == i)].values[:,:3]
            
            # N_k x K matrix with P(O_t,S=k|O_{t-1},theta_k), where O_t is data at time t
            probs = hmm.filter(tmp_data)

            # Product over t (rows) to get sequence prob (1 x K)
            # Each col is now P(O,S=k|theta_k) 
            res = probs[-1,:] #.prod(axis=0) + sys.float_info.min

            # This entry gives cross entropy between all states i=1,...,K and the true state k
            logprobs = np.log(res/res[k])
            try:
                CL[:,k] += logprobs / len(tmp_data)
            except ValueError:
                print(f'max_K: {max_K}')
                print(f'unique values: {values}')
                print(f'logprobs shape: {logprobs.shape}')
                print(f'shape res: {res.shape}')
                raise ValueError


    # Create plot of cross-likelihoods
    max_mag = np.max([np.abs(CL.min()),np.abs(CL.max())])

    min_val = -max_mag
    max_val = max_mag

    norm2 = Normalize(vmin=min_val,vmax=max_val)

    fig2 , ax = plt.subplots(1,1)
    im = ax.imshow(CL,norm=norm2,cmap='RdBu_r')
    cbar = fig2.colorbar(im,ax=ax)
    ax.set_title('Cross-Likelihoods')
    ax.set_ylabel('State')
    ax.set_xlabel('State')
    cbar.ax.set_ylabel('Nats', rotation=270)

    fig2.savefig(savename)

    # Create same but truncated plot
    CL_new = np.minimum(np.maximum(CL,-200),200)

    min_val_CL = CL.min()
    max_val_CL = CL.max()

    min_val = CL_new.min()
    max_val = CL_new.max()

    max_mag = np.max([np.abs(CL_new.min()),np.abs(CL_new.max())])

    min_val = -max_mag
    max_val = max_mag

    norm=Normalize(vmin=min_val, vmax=max_val)

    fig , ax = plt.subplots(1,1)
    im = ax.imshow(CL_new,norm=norm,cmap='RdBu_r')
    cbar = fig.colorbar(im,ax=ax)
    ax.set_title('Clamped Cross-Likelihoods')
    ax.set_ylabel('State')
    ax.set_xlabel('State')
    cbar.ax.set_ylabel('Nats', rotation=270)

    name , ext = os.path.splitext(savename)

    newname = name + '_clamped' + ext

    fig.savefig(newname)


def plot_state_by_pos(input_df,dlc_data,savename):
    pose_df = input_df.copy() 

    dlc_df = pd.read_csv(dlc_data,header=2)
    df = dlc_df[['x','y']]    
    df = df.iloc[pose_df['frame']/2]
    pose_df['x'] = df['x'].values
    pose_df['y'] = 448 - df['y'].values # Need to invert coords because video has (0,0) in top left

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
    cmap = gradient_cmap(colors,nsteps=len(np.unique(pose_df['state'])))
    vals = np.unique(pose_df['state'])
    m_colors = {f'{val/np.max(vals)}':rgb_to_hex(cmap(val/np.max(vals))[:-1]) for val in vals}
    
    pose_df['state'] = pose_df['state'].astype(str)

    fig = px.scatter(pose_df,x='x',y='y',color='state',size='speed',opacity=1,size_max=10)
    fig.update_traces(marker=dict(line=dict(width=0)))

    fig.write_html(savename)

    name , ext = os.path.splitext(savename)
    newname = name + '.pdf'

    #fig.write_image(newname)


def make_plots(data_csv, vid_name, raw_dlc_data, interval_start,interval_end,hmm_file=None, div=10,ARHMM_kwargs={'kappa':1,'use_best_K':True,'K':None,'epochs':20,'start_K':1,'end_K':30},make_vid=True,smooth=True,box_size=0.6,box_pixel=400,T=2e-3,d=5,seed=None,early_stopping=False,threshold=0.001):
    
    if not hmm_file == None:
        with open(hmm_file,'rb') as f:
            hmm = pickle.load(f)
            hmm_given = True

        arg_dict = {   'data_csv':data_csv, 
                        'vid_name':vid_name, 
                        'hmm':hmm_file,
                        'interval':f'range({interval_start},{interval_end})',
                        'div':div,
                        'make_vid':make_vid,
                        'smooth':smooth,
                        'box_size':box_size,
                        'box_pixel':box_pixel,
                        'seed':seed
                        }
    else:
        
        arg_dict = {   'data_csv':data_csv, 
                        'vid_name':vid_name, 
                        'interval':f'range({interval_start},{interval_end})',
                        'div':div,
                        'ARHMM_kwargs':ARHMM_kwargs,
                        'make_vid':make_vid,
                        'smooth':smooth,
                        'box_size':box_size,
                        'box_pixel':box_pixel,
                        'seed':seed,
                        'early_stopping':early_stopping,
                        'stopping_threshold':threshold}
                        
        hmm_given = False

    pbar = tqdm.tqdm(total=13)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    p = os.path.normpath(data_csv)
    newpath = os.sep.join([os.sep.join(p.split(os.sep)[:-1]), 'results',timestr])

    Path(newpath).mkdir(parents=True, exist_ok=True)

    kappa = ARHMM_kwargs['kappa']

    
    data_frame2 = pd.read_csv(data_csv)
    
    if interval_end in ['end',-1]:
        interval_end = len(data_frame2) - 1

    interval = range(interval_start,interval_end)

    # Preprocess data
    data_to_use , df , speed , head_body_angle , body_len , outliers = preprocess_data(data_frame2, interval, div, smooth, box_size, box_pixel, d, T)

    interval = np.linspace(interval_start + d,interval_end-len(outliers),len(speed))

    pbar.update(1)

    if not hmm_given:
        #¤¤ Produce ARHMM data! ¤¤¤
        #best_hmm , best_train_lls , best_val_lls , best_K_AIC , best_K_ll , val_lls , train_lls , AIC
        hmm , best_train_lls , best_val_lls , best_K , best_K_ll , val_lls , train_lls , AIC = produce_ARHMM_data(data_to_use, ARHMM_kwargs, seed, early_stopping=early_stopping,threshold=threshold)

        modelsave = get_savestr(newpath,f'hmm_model_{best_K}_states_{kappa}_kappa','.p')

        with open(modelsave,'wb') as f:
            pickle.dump(hmm,f)

        print("Saved hmm model!")
    else:
        best_K = hmm.K

    hmm_z = hmm.most_likely_states(data_to_use,)

    # Set colormap!
    
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

    colors_palette = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors_palette)

    values = np.unique(hmm_z)
    num_states = len(values)
    colors = [ cmap(value/np.max(values)) for value in range(values[0],values[-1]+1)]
    
    patches = [ mpatches.Patch(color=colors[i], label=f"State {k}") for i,k in enumerate(values) ]

    if not hmm_given:
        if ARHMM_kwargs['K'] == None:
            # Plot likelihoods
            # For all tried K
            
            plt.figure(1)
            plt.plot(np.arange(ARHMM_kwargs['start_K'],ARHMM_kwargs['end_K']),np.max(val_lls,axis=1))
            plt.xlabel("States K")
            plt.ylabel("Val log likelihood")
            plt.savefig(get_savestr(newpath,'val_ll_all','.png'))

            plt.figure(2)
            plt.plot(np.arange(ARHMM_kwargs['start_K'],ARHMM_kwargs['end_K']),np.max(train_lls,axis=1))
            plt.xlabel("States K")
            plt.ylabel("Train log likelihood")
            plt.savefig(get_savestr(newpath,'train_ll_all','.png'))
            
        # For best K
        plt.figure()
        plt.plot(best_val_lls)
        plt.xlabel("Epochs")
        plt.ylabel('Log likelihood')
        #plt.title(f'Best K={best_K} log likelihood evolution')
        plt.savefig(get_savestr(newpath,'val_ll_best','.png'))

        pbar.update(1)

    # Plot results
    xmin = interval[0]
    xmax = interval[-1]

    # Plot transition matrix
    learned_transition_mat = hmm.transitions.transition_matrix
    fig = plt.figure(figsize=(8, 4))
    im = plt.imshow(learned_transition_mat, cmap='gray')
    plt.title("Learned Transition Matrix")
    cbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.8]) 
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(get_savestr(newpath,'trans_mat','.png'))

    pbar.update(1)



    # Compute mean time for each state
    ticks = {k:[] for k in values}

    prev_state = 0
    tick = 0
    for state in hmm_z:
        if state == prev_state:
            tick+=1
            prev_state = state
        else:
            ticks[state].append(tick)
            tick = 0
            prev_state = state

    state_dur_means = list(map(lambda x : np.mean(x)*div, ticks.values()))
    #print(np.mean(state_dur_means))
    # Plot mean state durations
    plt.figure()
    plt.bar(values,state_dur_means,color=colors,label=['State' + str(s) for s in values])
    lgd = plt.legend(handles=patches,bbox_to_anchor=(1.05,1),loc=2,ncol=(len(values) // 11) + 1)
    plt.title('Mean state durations')
    plt.xlabel('State')
    plt.ylabel('Frames')
    plt.savefig(get_savestr(newpath,'mean_state_durs','.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')

    pbar.update(1)

    # Plot variables over time
    fig, [ax1, ax2, ax3 , ax4] = plt.subplots(4, 1, sharex=True,figsize=(16,12))
    
    ax1.plot(interval,speed)
    ax1.set_title('Speed')
    ax1.set_ylabel('m/s')
    
    ax2.plot(interval,head_body_angle)
    ax2.set_title('Head body angle')
    ax2.set_ylabel('Radians')

    ax3.plot(interval,body_len)
    ax3.set_title('Body length')
    ax3.set_ylabel('Meters')

    ax4.imshow(hmm_z[None,d:], aspect="auto", cmap=cmap, vmin=0, vmax=max(hmm_z),extent=[xmin,xmax,0,1],interpolation='none')
    ax4.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=(len(values) // 11) + 1)
    ax4.set_ylabel("$z_{\\mathrm{inferred}}$")
    ax4.set_xlabel("Frame")
    ax4.set_yticks([])

    plt.xlim(xmin, xmax)
    plt.savefig(get_savestr(newpath,'vars_states','.png'))

    pbar.update(1)
    
    # Plot histogram of state durations for each state
    inferred_state_list, inferred_durations = ssm.util.rle(hmm_z)

    # Rearrange the lists of durations to be a nested list where
    # the nth inner list is a list of durations for state n
    inf_durs_stacked = []
    for s in range(num_states):
        inf_durs_stacked.append(inferred_durations[inferred_state_list == s])

    fig , ax = plt.subplots(1,1)

    N , bins , hist_patches = ax.hist(inf_durs_stacked, label=['State ' + str(s) for s in range(num_states)])

    ax.set_xlabel('Duration (frames)')
    ax.set_ylabel('Frequency')
    lgd = ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=(len(values) // 11) + 1)
    ax.set_title('Histogram of Inferred State Durations')
    fig.savefig(get_savestr(newpath,'dur_hist','.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')

    pbar.update(1)


    # Make histogram plots for each state and each variable

    df['state'] = hmm_z

    # Plot states and speed overlayed on coordinates

    plot_state_by_pos(df,raw_dlc_data,get_savestr(newpath,'state_speed_coords','.html'))
    
    pbar.update(1)

    fig, ax = plt.subplots(num_states, 3,sharey=True,figsize=(20,30))

    for state_i in range(num_states):
        state_1 = [1 if num == state_i else 0 for num in hmm_z]
        
        ax[state_i,0].hist(speed[hmm_z == state_i],bins=50)
        ax[state_i,0].set_title(f'Speed state {state_i}')
        ax[state_i,0].set_ylabel('Frequency')
        
        ax[state_i,1].hist(head_body_angle[hmm_z == state_i],bins=50)
        ax[state_i,1].set_title(f'Yaw state {state_i}')
        
        ax[state_i,2].hist(body_len[hmm_z == state_i],bins=50)
        ax[state_i,2].set_title(f'Body length state {state_i}')

    ax[state_i,0].set_xlabel('m/s')
    ax[state_i,1].set_xlabel('Radians')
    ax[state_i,2].set_xlabel('Meters')

    plt.tight_layout()
    plt.savefig(get_savestr(newpath,'state_var_hist','.png'))

    pbar.update(1)
    
    # Make 3D scatter plot with all variables
    fig = px.scatter_3d(df, x='speed', y='bl', z='yaw',
                color='state',symbol='state',opacity=0.7)

    fig.update_traces(marker=dict(size=4))

    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                          ticks="outside"))

    fig.write_html(get_savestr(newpath,'3d_scatter_states','.html'))
    
    pbar.update(1)

    make_cross_likelihood_matrix(hmm,df,get_savestr(newpath,'cross_likelihood_mat','.png'))

    pbar.update(1)
    print("Plotting complete 3D trajectories...")
    make_hmm_3d_trajectories(df,get_savestr(newpath, f'3d_trajectories_{best_K}','.html'))

    pbar.update(1)

    print("Plotting separate 3D trajectories...")
    make_separate_3d_trajectories(df,get_savestr(newpath, f'3d_trajectory_state','.html'))

    pbar.update(1)

    # Make video with predicted state overlayed on original video
    if make_vid:
        print("Making video...")
        overlay_states(vid_name, data_csv, hmm_z, interval,savename=get_savestr(newpath,'state_vid',os.path.splitext(vid_name)[-1]))

        make_state_videos(df, vid_name, get_savestr(newpath, 'state_visits','.mp4'))

    txt_savefile = get_savestr(newpath,'variables','.txt')


    # Save all variables to txt file for inspection
    with open(txt_savefile,'w') as f:
        json_string = json.dumps(arg_dict,indent=2)
        f.write(json_string)

    pbar.update(1)




if __name__ == '__main__':

    # Frans vid
    #data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_pose_data.csv'
    #vid_name = rc:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'
    #dlc_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_frans\20201120_behaviour2020_v_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000.csv'
    

    # 2050 vid
    data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_pose_data.csv'
    vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'
    dlc_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2050\20201120_behaviour2020_v_2050_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000.csv'

    # 2053 vid
    #data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_pose_data.csv'
    #vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'
    #dlc_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000.csv'
    
    # 2053 interval
    #interval_start = 233694 - 220000
    #interval_end = 'end'

    #hmm_file = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\mouse_2050_20201216-105323\mouse_2050_20201216-105323\hmm_model_25_states_1_kappa.p'
    hmm_file = None

    interval_start = 0
    interval_end = 220000
    div = 10

    ARHMM_kwargs={  'kappa':2500,
                    'use_best_K':True, 
                    'K':11,
                    'epochs':20,
                    'start_K':1,
                    'end_K':30}

    make_vid = False
    smooth = True
    box_size = 0.6
    box_pixel = 400
    T = 2e-3
    d = 5
    seed = 1337
    early_stopping = True
    threshold = 0.001

    make_plots(data_csv, vid_name, dlc_csv, interval_start, interval_end, hmm_file, div, ARHMM_kwargs, make_vid, smooth, box_size, box_pixel, T, d,seed,early_stopping,threshold)