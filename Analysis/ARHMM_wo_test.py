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
def fit_ARHMM(train_data,val_data,obs_type='ar', obs_kwargs={'lags': 1},obs_init_kwargs = {'localize': True},transitions = 'stationary',transition_kwargs = None,num_states = 15,epochs = 20, seed=None,threshold=0.001,early_stopping=False):
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
    np.random.seed(seed)
    #print(f'Using early stopping: {early_stopping}')
    #print(obs_kwargs)
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
    hmm.observations.initialize(train_data)
    if early_stopping:
        min_val = -10e10 #probably small enough
        train_lls = min_val*np.ones(epochs-1)
        val_lls = min_val*np.ones(epochs-1)
    else:
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
            if abs(val_lls[epoch-2] - val_ll) < threshold and early_stopping and epoch > 10:
                print("Early stopped at epoch ", epoch)
                val_lls = val_lls[val_lls>min_val]
                train_lls = train_lls[train_lls>min_val]
                break
            #print("Trainling LL ", tr_ll, " Val LL ", val_ll)
    #print("Done")
    #print("K=",num_states,"test_ll=",test_ll)

    #test_ll = hmm.log_likelihood(test_data)/test_data.shape[0]
    zs = hmm.most_likely_states(train_data)
    usage = np.bincount(zs, minlength=hmm.K)
    perm = np.argsort(usage)[::-1]
    hmm.permute(perm)
    return train_lls, val_lls, hmm

def find_best_K(train_data,val_data,start_K,stop_K,kappa=1,seed=None,threshold=0.001,early_stopping=False,epochs=20,return_all=False):
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
    
    hmm_s = []
    #test_lls = np.zeros(stop_K-start_K)
    all_val_lls = np.zeros((stop_K-start_K,epochs))
    all_train_lls = np.zeros((stop_K-start_K,epochs))
    best_K = 0
    for K in trange(start_K,stop_K):
        #print("Training ARHMM with %d states.",K)
        if kappa==0:
            train_lls,val_lls, hmm = fit_ARHMM(train_data, val_data, epochs=epochs, num_states=K,seed=seed,threshold=threshold,early_stopping=early_stopping)
        else: #sticky HMM with kappa = int
            train_lls,val_lls, hmm = fit_ARHMM(train_data, val_data, epochs=epochs, num_states=K,transitions = 'sticky',transition_kwargs={'kappa': kappa},seed=seed,threshold=threshold,early_stopping=early_stopping)
        all_train_lls[K-start_K,:len(train_lls)] = train_lls
        all_val_lls[K-start_K,:len(val_lls)] = val_lls
        hmm_s.append(hmm)

    #print(all_val_lls)
    #print(np.amax(all_val_lls,axis=1))
    true_lls = val_data.shape[0]*np.amax(all_val_lls,axis=1)
    K_s = np.arange(1,all_val_lls.shape[0]+1)
    AIC = 2*K_s-2*true_lls

    hmm_s_index = np.argmax(true_lls)
    hmm_s_index_AIC = np.argmin(AIC)

    #hmm_s_index = np.argmin(AIC) if use_AIC else np.argmax(true_lls)
    
    """
    prev_last_5_means =0
    num_of_lags = 5
    K_s = np.zeros(stop_K-start_K)
    
    for i in range(num_of_lags,len(val_lls)):
        
        last_5_means = np.mean(val_lls[num_of_lags:i])
        #print(last_5_means)
        #print(last_5_means-prev_last_5_means)
        if last_5_means-prev_last_5_means < 0.001:
            K_s[i] = i
            print("<0.001",i)
        prev_last_5_means = last_5_means

    for i in range(num_of_lags,len(K_s)):
        try:
            if K_s[i] > 0 and K_s[i+1]-K_s[i] == 1 and K_s[i+2]-K_s[i+1] == 1 and K_s[i+3]-K_s[i+2]==1:
                best_K = K_s[i]
        except:
            best_K = K_s[-1]
    """

    
    
    best_K = hmm_s_index + start_K
    best_K_AIC = hmm_s_index_AIC + start_K
    best_hmm = hmm_s[hmm_s_index]
    best_hmm_AIC = hmm_s[hmm_s_index_AIC]

    if return_all:
        return  all_train_lls , all_val_lls , best_hmm_AIC , best_K_AIC , best_hmm , best_K , AIC
    else:
        return  all_train_lls , all_val_lls , best_hmm_AIC , best_K_AIC

def merge_two_states(hmm, s1, s2, datas, observations="ar"):
    
    def collapse_and_sum_2d(arr, i, j, axis=0):
        assert axis <= 1
        out = arr.copy()
        if axis == 0:
            out[i,:] += out[j,:]
            return np.delete(out, j, axis=0)
        if axis == 1:
            out[:, i] += out[:, j]
            return np.delete(out, j, axis=1)
        
    K = hmm.K
    D = hmm.D
    assert K >= 2
    assert s1 < K
    assert s2 < K
    assert s1 != s2
    datas = datas if isinstance(datas, list) else [datas]
    inputs, masks, tags = [None], [None], [None]

    expectations = [hmm.expected_states(data, input, mask, tag)
                            for data, input, mask, tag in zip(datas, inputs, masks, tags)]
    #print(expectations)
    # Merge expectations for 2 states
    expectations_new = []
    inputs = [np.array([np.array([])]*datas[0].shape[0])]
    masks = [np.ones((datas[0].shape[0], D))]
    tags = [None]
    for (Ez, Ezz, py) in expectations:
        T_curr = Ez.shape[0]
        
        # Put merged expectations in first column
        Ez_new = collapse_and_sum_2d(Ez, s1, s2, axis=1)
                
        # Now merge Ezz
        # Ezz will have shape 1, K, K
        # so we get rid of the first dimension then add it back.
        Ezz_new = collapse_and_sum_2d(Ezz[0], s1, s2, axis=0)
        Ezz_new = collapse_and_sum_2d(Ezz_new, s1, s2, axis=1)
        Ezz_new = Ezz_new[None, :, :]
        
        expectations_new.append((Ez_new, Ezz_new, py))
    
    # Perform M-Step to get params for new hmm
    new_hmm = ssm.HMM(K-1, D, observations=observations)
    #print(D)
    #obs_init_kwargs = {'localize': True}
    #new_hmm.initialize(data)
    #new_hmm.observations.initialize(data,**obs_init_kwargs)
    
    new_hmm.init_state_distn.m_step(expectations_new, datas, inputs, masks, tags)
    new_hmm.transitions.m_step(expectations_new, datas, inputs, masks, tags)
    #print(masks)
    new_hmm.observations.m_step(expectations_new, datas, inputs, masks, tags)
    
    # Evaluate log_likelihood
    expectations = [new_hmm.expected_states(data, input, mask, tag)
                    for data, input, mask, tag in zip(datas, inputs, masks, tags)]
    new_ll = new_hmm.log_prior() + sum([ll for (_, _, ll) in expectations])
    return new_ll, new_hmm

def make_similarity_matrix(hmm, data):
    num_states = hmm.K
    init_ll = hmm.log_probability(data)
    similarity = np.nan * np.ones((num_states, num_states))
    merged_hmms = np.empty((num_states, num_states), dtype=object)
    for s1 in range(num_states-1):
        for s2 in range(s1+1, num_states):
            merged_ll, merged_hmm = merge_two_states(hmm, s1, s2, data)
            similarity[s1, s2] = merged_ll - init_ll
            merged_hmms[s1, s2] = merged_hmm
            
    return similarity, merged_hmms

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