#!~/anaconda3/envs/DLC-GPU/bin/python
# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
import os
from ARHMM_wo_test import find_best_K , fit_ARHMM
from scipy.ndimage import gaussian_filter1d

def calc_speed(data,box_size=0.6,box_pixel=400,d=4,T=2e-3):
    # box_size = size of box in m
    # box_pixel = number of pixels in box
    # d = number of frames between speed
    # T = frame rate in seconds
    
    speed = np.zeros(data.shape[0]-d)
    for row_num in range(d,data.shape[0]):
        row = data[row_num,:]
        cog_x = data[row_num,4]
        cog_y = data[row_num,5]
        cog_x_prev = data[row_num-d,4]
        cog_y_prev = data[row_num-d,5]
        dT = d*T
        dxdt = (cog_x-cog_x_prev)/dT
        dydt = (cog_y-cog_y_prev)/dT
        speed[row_num-d] = box_size/box_pixel*np.sqrt(dxdt**2+dydt**2) #box 0.2 m and 400 px

    return speed


def autocorr2(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2
    return c[:len(x)//2]


def produce_ARHMM_data(data_to_use,ARHMM_kwargs={'kappa':1,'K':None,'epochs':20,'start_K':1,'end_K':30},seed=None,early_stopping=False,threshold=0.001,use_AIC=True,return_all=False):
    
    assert not((not ARHMM_kwargs['use_best_K']) and (ARHMM_kwargs['K'] is None)), 'Must specify K!'
    
    kappa = ARHMM_kwargs['kappa']
    
    # Search for best K
    start_K = ARHMM_kwargs['start_K']
    stop_K = ARHMM_kwargs['end_K']

    train_data, val_data, = train_test_split(data_to_use, test_size=0.3,shuffle=False)
    
    hmm_ll = []
    AIC = []
    hmm_AIC = None
    best_K_AIC = None
    best_K_ll = None
    val_lls = []
    train_lls = []

    if ARHMM_kwargs['K'] is None:
        print("Finding best K!")
        if return_all:
            train_lls, val_lls, hmm_AIC, best_K_AIC, hmm_ll, best_K_ll, AIC  = find_best_K(train_data, val_data,start_K,stop_K,kappa,seed,early_stopping=early_stopping,threshold=threshold,return_all=return_all)
        else:
            train_lls, val_lls, hmm_AIC, best_K_AIC = find_best_K(train_data, val_data,start_K,stop_K,kappa,seed,early_stopping=early_stopping,threshold=threshold,return_all=return_all)
            
        num_states = best_K_AIC if use_AIC else best_K_ll
        print(f"Found best K: {best_K_AIC}")
    else:
        num_states = ARHMM_kwargs['K']
        print(f"Using given K: {num_states}")
        best_K_AIC = num_states

        
    best_train_lls , best_val_lls , best_hmm = fit_ARHMM(  train_data,
                                            val_data,
                                            num_states=num_states,
                                            epochs=ARHMM_kwargs['epochs'],
                                            transitions='sticky',
                                            transition_kwargs={'kappa':kappa},
                                            seed=seed,
                                            early_stopping=early_stopping)
    
    
    return best_hmm , best_train_lls , best_val_lls , best_K_AIC , best_K_ll , val_lls , train_lls , AIC


def preprocess_data(df,interval,div, smooth=True, box_size=0.6, box_pixel=400, d=5, T=2e-3):    
    
    df = df.iloc[interval,:]
    data = df.to_numpy()
    newdf = df.iloc[d-1:,:]
    # Compute speed
    speed = calc_speed(data,box_size,box_pixel,d-1,T)
    body_len = box_size/box_pixel*data[d-1:,1]
    head_body_angle = data[d-1:,-1]

    N = len(speed[::div])

    newdf['speed'] = speed
    newdf['yaw'] = newdf['head_body_angle']

    indices_of_outliers1 = np.nonzero(head_body_angle > np.pi/2)
    indices_of_outliers2 = np.nonzero(head_body_angle < -np.pi/2)
    indices_of_outliers3 = np.nonzero(speed > 0.4)

    indices_of_outliers = np.append(indices_of_outliers1,indices_of_outliers2)
    indices_of_outliers = np.append(indices_of_outliers, indices_of_outliers3)
    
    speed = np.delete(speed,indices_of_outliers)
    body_len = np.delete(body_len,indices_of_outliers)
    head_body_angle = np.delete(head_body_angle,indices_of_outliers)

    outliers = np.concatenate((indices_of_outliers1[0], indices_of_outliers2[0], indices_of_outliers3[0]))
    outliers = np.unique(outliers)
    
    n_outliers = len(outliers)

    newdf = newdf[newdf['yaw'] < np.pi/2]
    newdf = newdf[newdf['yaw'] > -np.pi/2]
    newdf = newdf[newdf['speed'] < 0.4]

    # Smoot hdata and preprocess (reshape)
    wiki_const = 6.366 / div

    if smooth:
        speed = gaussian_filter1d(speed,wiki_const)
        body_len = gaussian_filter1d(body_len,wiki_const)
        head_body_angle = gaussian_filter1d(head_body_angle,wiki_const)

    speed = speed[::div]
    body_len = body_len[::div]
    head_body_angle = head_body_angle[::div]
    newdf = newdf.iloc[::div,:]

    newdf['yaw'] = head_body_angle
    newdf['bl'] = body_len
    newdf['speed'] = speed

    data_to_use = np.vstack((speed,head_body_angle))
    data_to_use = np.vstack((data_to_use, body_len)).T

    print(f"Number of outliers removed: {n_outliers} ({n_outliers/N} %)")

    return data_to_use , newdf , speed , head_body_angle , body_len , outliers 

def get_savestr(path, in_str , ext): 
    return os.sep.join([path, in_str + ext])


def rgb_to_hex(rgb,scaling=256):
    
    return f'#{int(rgb[0]*scaling):02x}{int(rgb[1]*scaling):02x}{int(rgb[2]*scaling):02x}'
