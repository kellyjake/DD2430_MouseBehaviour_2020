import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties

def triple_plot(df,xlab,to_plot,savename,events={},colors=None,figsize=(200,100)):
    mpl.rcParams['agg.path.chunksize'] = 20000

    if colors:
        assert(len(colors) == len(events))
        cols = colors
    else:
        cols = list(mcolors.BASE_COLORS.keys())[:len(events)]

    for col in mcolors.BASE_COLORS.keys():
        if col not in cols:
            data_col = col
            break
    fontP = FontProperties()
    fontP.set_size(40)
    print("Making subplots")
    fig = plt.figure(figsize=figsize)
    #fig, ax = plt.subplots(nrows=len(to_plot) + min(len(events),1),ncols=1,sharex=True,figsize=figsize)
    
    N = len(df)
    gs = gridspec.GridSpec(len(to_plot) + 1,1)

    for i,idx in enumerate(to_plot):
        ax = fig.add_subplot(gs[i,0])
        print(F"Plotting plot {i}")
        ax.plot(df[xlab],df[idx],linewidth=1,color=data_col)
        ax.grid(True)
        ax.set_xticks(np.arange(0,N+1,int(N/20)))
        ax.legend(bbox_to_anchor=(0,1),borderaxespad=5,prop=fontP)
        for j,key in enumerate(events.keys()):
            for line in events[key]:
                ax.axvline(line, linestyle='dashed',alpha=0.8,lw=1,color=cols[j])
    print("Done!")
    print("Plotting eventplot")
    if events:
        ax = fig.add_subplot(gs[-1,0])
        ax.eventplot(events.values(),linestyles='dashed',alpha=0.8,lw=1,colors=cols,linelengths=0.8)
        ax.set_yticks([])
        ax.grid(True)
        ax.legend(list(events.keys()),bbox_to_anchor=(0.,1.,0,0),ncol=1,borderaxespad=5,prop=fontP)
    print("Done!")
    print("Saving fig")
    fig.savefig(savename)


n = 1000
m = int(n/10)
a = np.random.rand(n)
b = np.random.rand(n)

laser1 = np.zeros((n))
laser2 = np.zeros((n))
X = np.linspace(1,n,n)
idx = np.random.choice(range(0,n),m,replace=False)
idx2 = np.random.choice(range(0,n),m,replace=False)
idx3 = np.random.choice(range(0,n),m,replace=False)

for i,j in zip(idx,idx2):
    laser1[i] = 1
    laser2[j] = 1

df = pd.DataFrame(data=X,columns=['X'])

df['Y'] = a
df['Z'] = b
df['Laser1'] = laser1
df['Laser2'] = laser2

idx_list = {'Laser1':df.index[df['Laser1'] == 1].tolist()}
idx_list['Laser2'] = df.index[df['Laser2'] == 1].tolist()
to_plot = ['Y','Z']

w=100
h=50

triple_plot(df,'X',to_plot,F'test_plot_{w}_{h}.png',idx_list,figsize=(w,h))

