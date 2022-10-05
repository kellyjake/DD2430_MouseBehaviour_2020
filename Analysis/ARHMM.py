import os
import time
import numpy as np
import random
import ssm
import pickle
from tqdm import tqdm , trange
import pandas as pd
from sklearn.model_selection import train_test_split

#obs = np.array()

#obs_type = 'ar'
#obs_kwargs = {'lags': 1} #?
#obs_init_kwargs = {'localize': True}
#transitions = 'stationary' #different with other kappa
#transition_kwargs = None #different with other kappa
#num_states = 10
#epochs = 10
def fit_ARHMM(train_data,val_data,test_data,obs_type='ar', obs_kwargs={'lags': 3},obs_init_kwargs = {'localize': True},transitions = 'stationary',transition_kwargs = None,num_states = 15,epochs = 20):
    """
    Fit an ARHMM on train_data and evaluate it on validation and test data.
    Args:
        train_data:
        val_data:
        test_data:
        obs_type: type of fitting AR in our case
        obs_kwargs: Number of lags 1 used by Behavenet
        obs_init_kwargs:
        transitions: How we transitions between states, stationary or sticky
                        With sticky it will be a bit harder to change state
        transition_kwargs: if transition=sticky then kwargs={'kappa':val}
        num_states: K value
        epochs: number of training epochs
    Returns:
        Log likelihood for training, validation and test set
        The HMM model
    """
    print(obs_kwargs)
    train_dim = train_data.shape[1]
    #print(train_dim)
    #print(train_data.shape)

    #if obs_kwargs['lags'] >  0:
    #obs_kwargs = {'lags': 1}
    #obs_init_kwargs = {'localize': True}
    #else:
    #   obs_kwargs = None
    #    obs_init_kwargs = {}

    hmm = ssm.HMM(
        num_states, train_dim,
        observations=obs_type, observation_kwargs=obs_kwargs,
        transitions=transitions, transition_kwargs=transition_kwargs)
    hmm.initialize(train_data)
    hmm.observations.initialize(train_data,**obs_init_kwargs)
    train_lls = np.zeros(epochs-1)
    val_lls = np.zeros(epochs-1)
    for epoch in range(epochs):
        #print('epoch %03i/%03i' % (epoch, epochs))
        if epoch > 0:
            hmm.fit(train_data, method='em', num_iters=1, initialize=False,verbose=1)


            tr_ll = hmm.log_likelihood(train_data) / train_data.shape[0]
            train_lls[epoch-1] = tr_ll
            val_ll = hmm.log_likelihood(val_data) / val_data.shape[0]
            val_lls[epoch-1] = val_ll
            #print("Trainling LL ", tr_ll, " Val LL ", val_ll)
    #print("Done")
    #print("K=",num_states,"test_ll=",test_ll)

    test_ll = hmm.log_likelihood(test_data)/test_data.shape[0]
    zs = hmm.most_likely_states(train_data)
    usage = np.bincount(zs, minlength=hmm.K)
    perm = np.argsort(usage)[::-1]
    hmm.permute(perm)
    return train_lls, val_lls, test_ll, hmm

def find_best_K(train_data,val_data,test_data,start_K,stop_K,kappa=10000):
    """
    Find the best K for an ARHMM on train_data and evaluate it on validation and test data.
    Args:
        train_data:
        val_data:
        test_data:
        start_K: lower limit of states
        stop_K: upper limit of states
    Returns:
        Log likelihood for training, validation and test set
        The HMM model
    """
    best_hmm = 0
    best_train_lls = 0
    best_val_lls = 0
    best_test_ll = 0
    test_lls = np.zeros(stop_K-start_K)
    all_train_lls = np.zeros(stop_K-start_K)
    best_K = 0
    for K in trange(start_K,stop_K):
        #print("Training ARHMM with %d states.",K)
        if kappa==0:
            train_lls,val_lls,test_ll, hmm = fit_ARHMM(train_data, val_data, test_data, epochs=20, num_states=K)
        else: #sticky HMM with kappa = int
            train_lls,val_lls,test_ll, hmm = fit_ARHMM(train_data, val_data, test_data, epochs=20, num_states=K,transitions = 'sticky',transition_kwargs={'kappa': kappa})
            
        all_train_lls[K-start_K] = np.amax(train_lls)
        test_lls[K-start_K] = test_ll
        if test_ll > best_test_ll:
            best_hmm = hmm
            best_train_lls = train_lls
            best_val_lls = val_lls
            best_test_ll = test_ll
            best_K = K

    return best_train_lls, best_val_lls, best_test_ll, test_lls,all_train_lls,hmm, best_K

"""
data_frame = pd.read_csv("pose_data_2d.csv")
data = data_frame.to_numpy()
#print(data)
box_size = 0.6 # size of box in m
box_pixel = 400 # number of pixels in box
d = 10 #number of frames between speed
T = 2e-3 # frame rate in seconds
speed = np.zeros(data.shape[0]-10)
for row_num in range(d,data.shape[0]):
    row = data[row_num,:]
    tail_x = data[row_num,5]
    tail_y = data[row_num,6]
    tail_x_prev = data[row_num-d,5]
    tail_y_prev = data[row_num-d,6]
    dT = d*T
    dxdt = (tail_x-tail_x_prev)/dT
    dydt = (tail_y-tail_y_prev)/dT
    speed[row_num-d] = box_size/box_pixel*np.sqrt(dxdt**2+dydt**2) #box 0.2 m and 400 px

body_len = box_size/box_pixel*data[:,1]
head_body_angle = data[:,-1]
data_to_use = np.array([speed,body_len[10:],head_body_angle[10:]]).T
start_K = 1
stop_K = 2
train_data, test_data, = train_test_split(data_to_use, test_size=0.3)
test_data, val_data = train_test_split(test_data, test_size=0.5)

best_train_lls, best_val_lls, best_test_ll, hmm, best_K =find_best_K(train_data, test_data, val_data,start_K,stop_K)
print("best train ll",best_train_lls)
print("best val ll",best_val_lls)
print("best test ll",best_test_ll)
"""