#!/usr/bin/env python
# coding: utf-8

# In[1]:


from extract_dataframe import extract_data
import ssm
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv , os , pickle , cv2 , tqdm , time
from ARHMM_wo_test import find_best_K , fit_ARHMM
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import ruptures as rpt
from overlay_states import overlay_states

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


# In[2]:


import sys , os
path = 'C:\\Users\\magnu\\OneDrive\\Dokument\\KI\\KI2020\\'

data_csv = os.path.join(path,"New_data_201120\\Mouse_frans\\20201120_behaviour2020_v_frans_1_pose_data.csv")
vid_name = os.path.join(path,"New_data_201120\\Mouse_frans\\20201120_behaviour2020_v_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4")


# In[3]:


interval = range(31135,37764)
data_frame2 = pd.read_csv(data_csv)
data = data_frame2.to_numpy()[interval,:]
print(data.shape)


# In[4]:


sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(data) # standardizing the data
pca = PCA()
X_pca = pca.fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[5]:


def calc_speed(data):
    box_size = 0.6 # size of box in m
    box_pixel = 400 # number of pixels in box
    d = 4 #number of frames between speed
    T = 2e-3 # frame rate in seconds
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


# In[6]:


div = 10


# In[7]:


# Remove outliers and compute speed

smooth = True
box_size = 0.6 # size of box in m
box_pixel = 400 # number of pixels in box
T = 2e-3 # frame rate in seconds
d = 5
speed = np.zeros(data.shape[0]-d)
speed = calc_speed(data)
speed = np.where(speed > 0.5,0.5,speed)

body_len = box_size/box_pixel*data[:,1]
body_len = np.where(body_len>0.7,0.7,body_len)[d-1:]

head_body_angle = np.rad2deg(data[:,-1])
head_body_angle = np.where(head_body_angle>40,40,head_body_angle)
head_body_angle = np.where(head_body_angle<-40,-40,head_body_angle)[d-1:]

print(np.amax(speed))
print(speed.shape)

speed = speed[::div]
body_len = body_len[::div]
head_body_angle = head_body_angle[::div]
print(speed.shape)
print(body_len.shape)
print(head_body_angle.shape)


# In[8]:


# Smoot hdata and preprocess (reshape)

wiki_const = 6.366

print(speed.shape)
if smooth:
    speed = gaussian_filter1d(speed,wiki_const / div)
    body_len = gaussian_filter1d(body_len,wiki_const / div)
    head_body_angle = gaussian_filter1d(head_body_angle,wiki_const / div)
data_to_use = np.vstack((speed,head_body_angle))
data_to_use = np.vstack((data_to_use, body_len))
#data_to_use = np.array([speed, head_body_angle[d-1:],body_len[d-1:]])
#data_to_use = data_to_use.reshape(3,780)
print(data_to_use.shape)
data_to_use = data_to_use.T
print(data_to_use.shape)
print(data_to_use.shape)
print(body_len[d:].shape)
print(speed.shape)


# In[9]:


start_K = 1
stop_K = 30
train_data, val_data, = train_test_split(data_to_use, test_size=0.3)
#test_data, val_data = train_test_split(test_data, test_size=0.5)
print(train_data.shape)
#best_train_lls, best_val_lls, best_test_ll, test_lls,train_lls,hmm, best_K =find_best_K(train_data, test_data, val_data,start_K,stop_K,2)
best_train_lls, best_val_lls, train_lls,val_lls,hmm, best_K = find_best_K(train_data, val_data,start_K,stop_K,1);
print("best train ll",best_train_lls)
print("best val ll",best_val_lls)
#print("best test ll",best_test_ll)
("")


# In[10]:


print("best_K",best_K)


# In[11]:


plt.figure(1)
plt.plot(np.arange(len(val_lls)),val_lls)
plt.xlabel("States K")
plt.ylabel("Val log likelihood")
plt.figure(2)
plt.plot(np.arange(len(train_lls)),train_lls)
plt.xlabel("States K")
plt.ylabel("Train log likelihood")


# In[ ]:





# In[12]:


num_states = 10

train_lls , val_lls , hmm = fit_ARHMM(train_data,val_data,num_states=num_states,epochs=15,transitions='sticky',transition_kwargs={'kappa':1})


# In[13]:


plt.plot(val_lls)


# In[14]:


import matplotlib.patches as mpatches

# Show transition matrix
learned_transition_mat = hmm.transitions.transition_matrix
fig = plt.figure(figsize=(8, 4))

im = plt.imshow(learned_transition_mat, cmap='gray')
plt.title("Learned Transition Matrix")

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

# Plot events
#print(max(hmm_lls))
time_bins = data_to_use.shape[1]
hmm_z = hmm.most_likely_states(data_to_use,)
plt.figure(1,figsize=(16, 2))
im = plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=max(hmm_z) + 1)
values = np.unique(hmm_z)
colors = [ im.cmap(im.norm(value)) for value in values]

patches = [ mpatches.Patch(color=colors[i], label="State {l}".format(l=values[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.xlim(0,10)
plt.ylabel("$z_{\\mathrm{inferred}}$")
#plt.yticks([])
plt.xlabel("time")


# Compute mean time for each state
ticks = []
prev_state = 0
tick = 0
for state in hmm_z:
    if state == prev_state:
        tick+=1
        prev_state = state
    else:
        ticks = np.append(ticks,tick)
        tick = 0
        prev_state = state
#print(hmm_z)
print(np.mean(ticks))


"""
time_bins = data_to_use.shape[1]
hmm_z = hmm_2.most_likely_states(data_to_use)
plt.subplot(212)
plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
#plt.xlim(2600, 2800)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")
"""

xmin = 0
xmax = int(data.shape[0]/div)

start_m = np.floor(data[0,0]/(130*60))
start_s = data[0,0]/130%60
stop_m = np.floor(data[-1,0]/(130*60))
stop_s = data[-1,0]/130%60
#print("start=",start_m,":",start_s,"stop=",stop_m,":",stop_s)
#plt.plot(np.arange(40,50),lls_40)
plt.figure(2,figsize=(16, 2))
plt.plot(np.arange(len(speed)),speed)
plt.title('Speed')
plt.xlim(0, xmax)

plt.figure(4,figsize=(16, 2))
plt.plot(np.arange(len(head_body_angle[d:])),head_body_angle[d:])
plt.title('Head body angle')
plt.xlim(0, xmax)

plt.figure(3,figsize=(16, 2))
plt.plot(np.arange(len(body_len[d:])),body_len[d:])
plt.xlim(0, xmax)
plt.title('Body length')


fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
ax1.xcorr(speed, body_len, usevlines=True, maxlags=50, normed=True, lw=2)
ax1.grid(True)

ax2.acorr(speed, usevlines=True, normed=True, maxlags=50, lw=2)
ax2.grid(True)

plt.show()
#true_state_list, true_durations = ssm.util.rle(true_states)

inferred_state_list, inferred_durations = ssm.util.rle(hmm_z)


# Rearrange the lists of durations to be a nested list where
# the nth inner list is a list of durations for state n
#true_durs_stacked = []
inf_durs_stacked = []
for s in range(num_states):
    #true_durs_stacked.append(true_durations[true_state_list == s])
    inf_durs_stacked.append(inferred_durations[inferred_state_list == s])

plt.figure(5)
fig = plt.figure(figsize=(8, 4))
plt.hist(inf_durs_stacked, label=['state ' + str(s) for s in range(num_states)])
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Histogram of Inferred State Durations')

plt.show()


# In[15]:


overlay_states(vid_name, data_csv, hmm_z, interval[::div])


# In[16]:


df = pd.DataFrame(data={'state':hmm_z,'speed':speed,'bl':body_len,'yaw':head_body_angle})


# In[17]:


plt.figure(figsize=(10,30))

corrmat = np.zeros((3,num_states))
for state_i in range(num_states):
    state_1 = [1 if num ==state_i else 0 for num in hmm_z]
    corrmat[0,state_i] = np.corrcoef(state_1,speed)[1,0]
    #print("Correlation with speed",corrmat[state_i,0])
    corrmat[1,state_i] = np.corrcoef(state_1,head_body_angle)[1,0]
    #print("Correlation with hba",corrmat[state_i,1])
    corrmat[2,state_i] = np.corrcoef(state_1,body_len)[1,0]
    #print("Correlation with bl",corrmat[state_i,2])
    
    plt.subplot(num_states, 3, state_i*3+1)
    plt.hist(speed[hmm_z == state_i],bins=50)
    plt.title(f'Speed state {state_i}')
    
    plt.subplot(num_states, 3, state_i*3+2)
    plt.hist(head_body_angle[hmm_z == state_i],bins=50)
    plt.title(f'Yaw state {state_i}')
    
    plt.subplot(num_states, 3, state_i*3+3)
    plt.hist(body_len[hmm_z == state_i],bins=50)
    plt.title(f'Body length state {state_i}')


# In[18]:



plt.figure()

plt.imshow(corrmat,cmap='Greys')
plt.colorbar()
plt.title('Correlation coefficients')
plt.xlabel('State')
plt.ylabel('Variable')


# In[37]:


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.express as px
from IPython.display import HTML , display
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[38]:




fig = px.scatter_3d(df, x='speed', y='bl', z='yaw',
              color='state',symbol='state',opacity=0.7)

HTML(fig.to_html())
fig.show()


# In[40]:




fig = px.scatter_3d(df, x='speed', y='bl', z='yaw',
              color='state',symbol='state',opacity=0.7)

HTML(fig.to_html())

