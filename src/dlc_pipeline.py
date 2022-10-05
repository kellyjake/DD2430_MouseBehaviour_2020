import numpy as np
import pandas as pd

import sys , os , argparse , math
from dlc_analysis import PoseExtractor

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm , gridspec
from matplotlib.font_manager import FontProperties
from tensorflow import ConfigProto , Session
import tkinter as tk
from tkinter import messagebox , filedialog
from tqdm import trange

def dlc_pipeline(video_filename,events_filename,config_path,max_frames,use2d=True,startframe=None,endframe=None):
    """
    This is a function used to extract pose kinematics from video_filename and plot them together with the  
    events taking place in events_filename.

    :param video_filename:  name of video to be analyzed
    :param events_filename: name of csv file with events from arena/arduino to be overlayed
    :param config_path:     path to config file of trained dlc model to be used to extract mouse position from video
    :param max_frames:      upper limit on how many frames the computation of baseline should consider (more -> slower but better baseline)
    :param startframe:      if given together with endframe these frames will be used for creating the baseline (for debug mainly)
    """

    pose = PoseExtractor(video_filename,config_path,use2d)
    
    print("PoseExtractor created")

    if not use2d:
        pose.create_baseline(video_filename,config_path,start_frame=startframe,end_frame=endframe)
        print("Baseline created")
    else:
        pose.create_videos()

    savefile_binary , savefile_csv = pose.extract_features(save_as_csv=True)
    print("Exctracted poses")
    events = extract_events(events_filename)
    print("Extracted events")
    create_plots(savefile_binary,events,max_frames)
    print("Plots created")
    
def query_ids_states(valid_IDs):
    """
    Creates window to parse the user which events should be overlayed with features in plot

    :param valid_IDs:   dictionary - keys: names of events - values: list of possible states for each event

    Example valid_IDs = {'Laser2':[0,1],'TrialBox':[0,1,2]} 

    :return:            dictionary of by user chosen variables and states
    """

    root = tk.Tk()
    root.title("Select variables and states to plot")
    root.withdraw()
    messagebox.showinfo("Choose variables and states to plot", message="Indicate which variables and states should be included in the plot by checking the checkbox.")
    
    root.deiconify()
    
    intvars = {}

    lab1 = tk.Label(root,text='Variables')
    lab1.grid(row=0,column=0)

    lab2 = tk.Label(root,text='Possible states')
    lab2.grid(row=0,column=1)


    for i,key in enumerate(valid_IDs.keys()):
        intvar = tk.IntVar()
        intvars[key] = {"var":intvar}
        cb = tk.Checkbutton(root, text = key, variable = intvars[key]["var"], onvalue = 1, offvalue = 0)
        cb.grid(row=i+1,column=0)
        intvars[key]["states"] = {}
        for j,state in enumerate(valid_IDs[key]):
            intvar = tk.IntVar()
            intvars[key]["states"][state] = intvar
            cb_sub = tk.Checkbutton(root, text = state, variable = intvars[key]["states"][state], onvalue = 1, offvalue = 0)
            cb_sub.grid(row=i+1,column=j+1)
    
    exit_btn = tk.Button(root,text='OK',command=root.destroy)
    exit_btn.grid(row=i+2,column=0)
    root.mainloop()

    chosen_vars = {}
    
    for key in intvars.keys():
        if intvars[key]["var"].get() == 1:
            chosen_vars[key] = []
            for state in intvars[key]["states"]:
                if intvars[key]["states"][state].get() == 1:
                    chosen_vars[key].append(state)

    root.quit()

    return chosen_vars


def extract_events(events_filename):
    """
    Function taking a csv file produced from arduino/arena and returns a dict with all events and their respective observed states

    :param events_filename: name of csv file containing events

    :return:                dictionary with event names as keys and list of observed states as values
    """

    print(F"Extracting events from {events_filename}...")

    df = pd.read_csv(events_filename)
    valid_IDs = {}

    for i, row in df.iterrows():
        this_id = row['ID']
        if not this_id in valid_IDs.keys():
            valid_IDs[this_id] = []

        state = row['State']
        if not state in valid_IDs[this_id]:
            valid_IDs[this_id].append(state)

    ids = query_ids_states(valid_IDs)

    query_dict = {}

    for key in ids.keys():
        query_dict[key] = {}
        for state in ids[key]:
            indexes = df.index[(df['ID'] == key) & (df['State'] == state)].tolist()
            vals = [df.iloc[index]['Time (ms)'] for index in indexes]
            query_dict[key][state] = vals

    print(F"Extracted events!")

    return query_dict


def create_plots(savefile_binary,events,max_frames):
    """
    Produces three plots from feature and event data:
    1,2: Center of gravity movement in arena over time, one with dots and one with lines (why both?)
    3: Plot of pitch, roll and yaw over time, with occuring events overlayed.
    The tripleplot (3) requires very high resolution by Dimitrios, so it has to be a very large figsize.
    This is not possible for very long videos, so the data is divided into multiple plots (as specified by max_frames).

    :param savefile_binary: path to saved extracted features (..._pose.npy file)
    :param events:          dictionary with keys being names of events to plot and values list of states for the given event (key) to plot
    :param max_frames:      max length of video before dividing into multiple plots
    """

    split = savefile_binary.split(os.path.sep)
    savepath = os.path.sep.join(split[:-1])

    pose = PoseExtractor(None,None)
    df = pose.create_df(savefile_binary)

    N = len(df)

    no_splits = N // max_frames

    print(F"Splitting into {no_splits + 1} plots.")
    print('Creating plots...')

    for i in trange(no_splits+1):
        tmp_df , tmp_events = partition_data(df,events,max_frames,i)
        draw_plots(tmp_df,savepath,i)

        savestr = 'gaze' + ''.join([F'_{key}' for key in events.keys()])

        gaze = ['head_angle','head_body_angle']
        triple_plot(tmp_df,'frame',gaze,os.path.join(savepath,F'{savestr}({i+1}).png'),events=tmp_events)

    print(f"Plots saved in {os.path.join(savepath,savestr)}(0-{i+1}).png")
        
def partition_data(df,events,max_N,i):
    """
    Returns a subset of data from pose data (df) and events given max_N

    :param df:      pandas dataframe with pose data (from ...pose.npy)
    :param events:  dictionary with keys being names of events to plot and values list of states for the given event (key) to plot
    :param max_N:   max length of dataframe before dividing into multiple splits
    :param i:       index of current subset

    :return:        a partitioned dataframe of pose data and dict of observed events and states in this timeframe
    """
    
    tmp_min_frame = i*max_N
    tmp_max_frame = (i+1)*max_N

    if tmp_max_frame >= len(df):
        tmp_df = df.iloc[tmp_min_frame:]
    else:
        tmp_df = df.iloc[tmp_min_frame:tmp_max_frame]
    
    tmp_min_val = tmp_df.iloc[0]['frame'] / 2
    tmp_max_val = tmp_df.iloc[-1]['frame'] / 2

    #print(tmp_min_val)
    #print(tmp_max_val)

    tmp_events = {}

    for key in events:
        tmp_events[key] = {}
        for state in events[key]:
            tmp_events[key][state] = []
            for val in events[key][state]:
                if tmp_min_val <= val <= tmp_max_val:
                    tmp_events[key][state].append(val)

    return tmp_df , tmp_events

def draw_plots(df,savepath,i):
    """
    Draws and saves the plots specified in create_plots

    :param df:          dataframe with pose data to plot
    :param savepath:    path to save files
    :param i:           index of plot (given multiple splits of data)
    """

    cmap = cm.get_cmap('Spectral')

    df.plot(x = 'cog_x', y='cog_y', kind='scatter', c='frame', cmap=cmap, s=1, alpha=.5)
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    plt.savefig(os.path.join(savepath,'cog_loc_i.png'))

    df.plot(x = 'cog_x', y='cog_y', kind='line')
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(savepath,'cog_line_i.png'))
    

def triple_plot(df,xlab,to_plot,savepath,events={},colors=None,figsize=(300,50),labelsize=20,fontsize=40):
    """
    Plots 3 as specified in draw_plots.

    :param df:      pandas dataframe with extracted pose data
    :param xlab:    label for x-axis to show
    :param to_plot: list of names (strings) of columns in dataframe to plot
    :param savepath:    path to save plot
    :param events:      dictionary with events and states to plot in eventplot
    :param colors:      list of colors to choose plot colors manually (otherwise chosen automatically)
    :param figsize:     size of figure (cm)
    
    Larger figsize than (300,50) will likely cause crash. This has been chosen specifically to give a good resolution for Dimitrios
    to inspect the plot manually.
    """
    
    # Helps when creating very large figures. Otherwise matplotlib crashes. 20000 somewhat arbitrarily set so that it works.
    # Increasing this will alleviate computations but might create artifacts in plot.
    mpl.rcParams['agg.path.chunksize'] = 20000

    # Mute non important pandas warning
    pd.options.mode.chained_assignment = None  # default='warn'

    event_dict = {}

    # Rename keys for legend
    for key in events.keys():
        for state in events[key]:
            new_key = F"{key}_{state}"
            event_dict[new_key] = events[key][state]

    #print(event_dict)

    # If user has not specified colors, pick colors yourself
    if colors:
        assert(len(colors) == len(event_dict))
        cols = colors
    else:
        cols = list(mcolors.TABLEAU_COLORS.keys())[:len(event_dict)]
    
    
    # Compute speed
    box_size = 0.6
    box_pixel = 400
    d = 5
    T = 2e-3
    speed = calc_speed(df.to_numpy(),box_size, box_pixel, d, T)

    df = df.iloc[d:,:]

    # Set font for ylabels
    fontP = FontProperties()
    fontP.set_size(fontsize)
    tick_frequency = 25
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(to_plot) + 2,1)
    tick_start = int(df.iloc[0]['frame']/2)
    tick_end = int(df.iloc[-1]['frame']/2)

    axes = []
    
    x_tick_interval = np.arange(tick_start,tick_end,int((tick_end - tick_start)/tick_frequency))

    for i,idx in enumerate(to_plot):
        if i > 0:
            ax = fig.add_subplot(gs[i,0],sharex=axes[0])
            ax.set_xlim([tick_start,tick_end])
        else:
            ax = fig.add_subplot(gs[i,0])

        axes.append(ax)
        
        if idx in ['head_angle','head_body_angle']:
            df[idx] = np.rad2deg(df[idx])

        ax.plot(df[xlab] / 2,df[idx],'.',markersize=1,color='k')
        ax.grid(True)
        #ax.set_xlabel('Milliseconds')
        ax.set_ylabel(f'{idx}')
        ax.set_xticks([])

        ax.tick_params(axis='x',labelsize=labelsize)

        #ax.legend(bbox_to_anchor=(0,1),borderaxespad=5,prop=fontP)
        # Want lines from eventplot to extend into all upper plots
        for j,key in enumerate(event_dict.keys()):
            for line in event_dict[key]:
                ax.axvline(line, linestyle='dashed',alpha=0.8,lw=1,color=cols[j])

    # Plot speed
    ax = fig.add_subplot(gs[i+1,0],sharex=axes[0])
    axes.append(ax)
    ax.plot(df[xlab] / 2 , speed, '.', markersize=1,color='k')
    ax.grid(True)
    #ax.set_xlabel('Milliseconds')
    ax.set_ylabel('Speed (m/s)')
    ax.set_xticks([])
    ax.set_ylim([0,4])
    ax.tick_params(axis='x',labelsize=labelsize)

    #ax.legend(bbox_to_anchor=(0,1),borderaxespad=5,prop=fontP)
    
    # Want lines from eventplot to extend into all upper plots
    for j,key in enumerate(event_dict.keys()):
            for line in event_dict[key]:
                ax.axvline(line, linestyle='dashed',alpha=0.8,lw=1,color=cols[j])


    # Eventplot is a method used specifically for plotting neuroscience-like events.
    if events:
        #print(events)
        ax = fig.add_subplot(gs[i+2,0],sharex=axes[0])
        ax.eventplot(event_dict.values(),linestyles='dashed',lw=1,colors=cols,linelengths=0.8)
        ax.set_yticks([])
        ax.grid(True)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.legend(list(event_dict.keys()),bbox_to_anchor=(0.,1.,0,0),ncol=1,borderaxespad=5,prop=fontP)
        ax.tick_params(axis='x',labelsize=labelsize)

    ax.set_xticks(x_tick_interval)
    labs = ax.get_xticklabels()
    newlabs = [ms2str(lab) for lab in x_tick_interval]
    #print(newlabs)
    ax.set_xticklabels(newlabs)
    

    print("Saving figure...")
    fig.savefig(savepath)

    plt.close(fig)
    

def ms2str(millisec_str):
    try:
        millisec = int(millisec_str)
    except ValueError:
        return millisec_str

    s = millisec / 1000
    m = int(s // 60)
    rem = round(s % 60,3)
    
    return f'{m} m , {rem} s'

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


def main(video_filename,event_filename,config_path,framestart,frameend,do_create_plots,binary_data,max_frames,use2d):
    root = tk.Tk()
    root.withdraw()

    if not video_filename and not do_create_plots:
        video_filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select video file", filetypes = [("Video file",["*.avi","*.mp4"])])
        print(f"Got video filename {video_filename}")

    if not event_filename:
        cwd = os.sep.join(os.path.splitext(video_filename)[0][:-1])
        event_filename = filedialog.askopenfilename(initialdir = cwd, title = "Select data file", filetypes = [("csv file","*.csv")])
        print(f"Got csv filename {event_filename}")
    
    if do_create_plots:
        if not binary_data:
            binary_data = filedialog.askopenfilename(initialdir = cwd, title = "Select binary pose data file", filetypes = [("pickle file","*.npy")])
        
        root.destroy() # DONT USE root.quit() HERE
        df = extract_events(event_filename)
        #print(df)
        create_plots(binary_data,df,max_frames)
    else:
        dlc_pipeline(video_filename,event_filename,config_path,max_frames,use2d,framestart,frameend)
    
    

if __name__ == "__main__":

    config1 = ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = Session(config=config1)
    
    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()

    CLI.add_argument(
        "--framestart",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=None,  # default if nothing is provided
    )
    CLI.add_argument(
        "--frameend",
        type=int,  # any type/callable can be used here
        default=None,
    )
    CLI.add_argument(
        "--config_path",
        type=str,
        default='/home/titan/KI2020/DLC/KI2020_Training-Valter-2020-09-19/config.yaml'
    )
    CLI.add_argument(
        "--video",
        type=str,
        default=''
    )
    CLI.add_argument(
        "--only_plots",
        type=bool,
        default=False
    )
    CLI.add_argument(
        "--binary_data",
        type=str
    )
    CLI.add_argument(
        "--events",
        type=str,
        default=''
    )
    CLI.add_argument(
        "--max_frames",
        type=int,
        default=100000
    )
    CLI.add_argument(
        "--use2d",
        type=int,
        default=1
    )

    # parse the command line
    args = CLI.parse_args()
    # access CLI options

    config_path = args.config_path
    video_filename = args.video
    event_filename = args.events
    framestart = args.framestart
    frameend = args.frameend
    makeplots = args.only_plots
    binary_data = args.binary_data
    max_frames = args.max_frames
    use2d = bool(args.use2d)

    main(video_filename,event_filename,config_path,framestart,frameend,makeplots,binary_data,max_frames,use2d)