from blink_sync import Syncer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def zscore_man(df):
    mean = np.mean(df)
    std = np.std(df)
    scores = (df - mean)/std
    
    return scores


videoFileName = r'e:\ExperimentalResults\20200727\20200727_behaviour2020_v_4\20200727_behaviour2020_v_4_recording_timed.avi'
timestamps = r'e:\ExperimentalResults\20200727\20200727_behaviour2020_v_4\20200727_behaviour2020_v_4_timestamps.csv'
dataFile = r'e:\ExperimentalResults\20200727\20200727_behaviour2020_v_4\20200727_behaviour2020_v_4_data.csv'

df = pd.read_csv(open(dataFile,'r'))
df2 = df.filter(['ID','Value'])
df3 = df2.loc[df2['ID'] == 'TrialLED']
df4 = df3.loc[df3['Value'] == 1]
len(df4)

syncer = Syncer(videoFileName,timestamps)
syncer.compute_intensities()

intens = np.array(syncer.get_intensities())
t = np.array(syncer.get_timestamps())

z = np.abs(zscore_man(intens))
    
thresh = 3

no_outliers = intens[(z < thresh)]

minus_backgr = no_outliers - no_outliers[0]

part_data = no_outliers[:10000]

plt.plot(range(len(part_data)),part_data)
plt.show()
data_next = part_data[1:]
data_prev = part_data[:-1]

diff = data_next - data_prev

plt.plot(range(len(diff)),diff)
plt.show()
