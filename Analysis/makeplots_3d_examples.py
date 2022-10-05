from ARHMM_plots import *

data_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_pose_data.csv'
vid_name = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000_labeled.mp4'
dlc_csv = r'c:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2430_-_Project_Course_in_Data_Science\sharing\New_data_201120\Mouse_2053\20201120_behaviour2020_v_2053_for_frans_1_recordingDLC_resnet50_KI2020_TrainingSep19shuffle1_400000.csv'
interval_start = 233694 - 220000
interval_end = 'end'
div = 10
threshold = 0.001


ARHMM_kwargs={  'kappa':1,
                'use_best_K':True, 
                'K':39,
                'epochs':50,
                'start_K':1,
                'end_K':50}

make_vid = True
smooth = True
box_size = 0.6
box_pixel = 400
T = 2e-3
d = 5
seed = 1337
early_stopping = True


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
print("Preprocessing data")
data_to_use , df , speed , head_body_angle , body_len , outliers = preprocess_data(data_frame2, interval, div, smooth, box_size, box_pixel, d, T)

interval = np.linspace(interval_start + d,interval_end-len(outliers),len(speed))

#¤¤ Produce ARHMM data! ¤¤¤
print("Producing ARHMM data")
#best_hmm , best_train_lls , best_val_lls , best_K_AIC , best_K_ll , val_lls , train_lls , AIC
hmm , best_train_lls , best_val_lls , best_K , best_K_ll , val_lls , train_lls , AIC = produce_ARHMM_data(data_to_use, ARHMM_kwargs, seed, early_stopping=early_stopping,threshold=threshold)

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
#print(colors)
#print(values)
patches = [ mpatches.Patch(color=colors[i], label=f"State {k}") for i,k in enumerate(values) ]

df['state'] = hmm_z

df['state_consec'] = (df.state.diff(1) != 0).astype('int')

df['state_appearence'] = df.groupby('state')['state_consec'].cumsum()

values = np.unique(df['state'])

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
cmap = gradient_cmap(colors)

state_colors = [rgb_to_hex(cmap(value/np.max(values))[:-1]) for value in values]

# Make 3D scatter plot with all variables
ranges = {
    'xmax':max(df['speed']),
    'xmin':min(df['speed']),
    'ymax':max(df['bl']),
    'ymin':min(df['bl']),
    'zmax':max(df['yaw']),
    'zmin':min(df['yaw'])}

titles = {'x':'Speed (m/sec)' ,
        'y':'Body length (m)',
        'z':'Yaw (degrees)'}

n = len(values)

cmap_trace = cm.get_cmap('CMRmap')

max_occ = []
bins = 50

for i,k in enumerate([0]):
    
    fig = go.Figure()

    newdf = df[df['state'] == k]
    occurences = np.unique(newdf['state_appearence'])

    #make_histogram_projections(fig,newdf,ranges,curr_row,curr_col,bar_color=state_colors[i])
    
    bar_color = state_colors[i]
    
    # data binning and traces
    col = 'bl'
    a0_bl=np.histogram(newdf[col], bins=bins, density=False)[0].tolist()
    a0_bl=np.repeat(a0_bl,2).tolist()
    a0_bl.insert(0,0)
    a0_bl.pop()
    a0_bl[-1]=0
    a1_bl=np.histogram(newdf[col], bins=bins-1, density=False)[1].tolist()
    a1_bl=np.repeat(a1_bl,2)

    verts, tri = triangulate_histogram([ranges['zmin']]*len(a0_bl), a1_bl, a0_bl / np.max(a0_bl))
    x_bl, y_bl, z_bl = verts.T
    I_bl, J_bl, K_bl = tri.T


    col = 'yaw'
    a0_yaw=np.histogram(newdf[col], bins=bins, density=False)[0].tolist()
    a0_yaw=np.repeat(a0_yaw,2).tolist()
    a0_yaw.insert(0,0)
    a0_yaw.pop()
    a0_yaw[-1]=0
    a1_yaw=np.histogram(newdf[col], bins=bins-1, density=False)[1].tolist()
    a1_yaw=np.repeat(a1_yaw,2)

    verts, tri = triangulate_histogram([ranges['xmin']]*len(a0_yaw), a1_yaw, a0_yaw / np.max(a0_yaw))
    x_yaw, y_yaw, z_yaw = verts.T
    I_yaw, J_yaw, K_yaw = tri.T

    col = 'speed'

    a0_speed=np.histogram(newdf[col], bins=bins, density=False)[0].tolist()
    a0_speed=np.repeat(a0_speed,2).tolist()
    a0_speed.insert(0,0)
    a0_speed.pop()
    a0_speed[-1]=0
    a1_speed=np.histogram(newdf[col], bins=bins-1, density=False)[1].tolist()
    a1_speed=np.repeat(a1_speed,2)

    verts, tri = triangulate_histogram([ranges['ymin']]*len(a0_speed), a1_speed, a0_speed / np.max(a0_speed))
    x_speed, y_speed, z_speed = verts.T
    I_speed, J_speed, K_speed = tri.T

    newranges = {
        'xmin':np.min(newdf['speed']),
        'xmax':np.max(newdf['speed']),
        'ymin':np.min(newdf['bl']),
        'ymax':np.max(newdf['bl']),
        'zmin':np.min(newdf['yaw']),
        'zmax':np.max(newdf['yaw'])
                }

    # Speed
    fig.add_traces(go.Mesh3d(x=newranges['xmin'] + (y_speed - np.min(y_speed))/(np.max(y_speed) - np.min(y_speed))*(newranges['xmax'] - newranges['xmin']), 
                             y=ranges['ymin'] + z_speed/np.max(z_speed)*(ranges['ymax'] - ranges['ymin'])*0.5, 
                             z=x_speed/np.max(x_speed)*ranges['zmin'], 
                             i=I_speed, j=J_speed, k=K_speed, color=bar_color, opacity=0.7))

    # Body length
    fig.add_traces(go.Mesh3d(x=x_bl/np.max(x_bl)*ranges['xmin'], 
                             y=newranges['ymin'] + (y_bl - np.min(y_bl))/(np.max(y_bl) - np.min(y_bl))*(newranges['ymax'] - newranges['ymin']), 
                             z=ranges['zmin'] + z_bl/np.max(z_bl)*(ranges['zmax'] - ranges['zmin'])*0.5, 
                             i=I_bl, j=J_bl, k=K_bl, color=bar_color, opacity=0.7))

    # Yaw
    fig.add_traces(go.Mesh3d(x=ranges['xmin'] + z_yaw/np.max(z_yaw)*(ranges['xmax'] - ranges['xmin'])*0.5, 
                             y=ranges['ymin']*x_yaw/np.max(x_yaw), 
                             z=newranges['zmin'] + (y_yaw - np.min(y_yaw))/(np.max(y_yaw) - np.min(y_yaw))*(newranges['zmax'] - newranges['zmin']), 
                             i=I_yaw, j=J_yaw, k=K_yaw, color=bar_color, opacity=0.7))

    if len(occurences) > 0:
        max_occ.append(max(occurences))
    else:
        max_occ.append(0)

    for occurence in occurences:
        tmpdf = newdf[newdf['state_appearence'] == occurence]
        
        colors = [ cmap_trace(j/len(tmpdf)) for j in range(len(tmpdf)) ]

        fig.add_trace(go.Scatter3d(
            x=tmpdf['speed'],
            y=tmpdf['bl'],
            z=tmpdf['yaw'],
            name = f'O:{occurence}',
            mode= 'lines',
            showlegend = False,
            line = dict(color=colors,width=4),
            ))
    
        # Add start and end points to each trajectory
        fig.add_trace(go.Scatter3d(
            x=[tmpdf['speed'].iloc[0]],
            y=[tmpdf['bl'].iloc[0]],
            z=[tmpdf['yaw'].iloc[0]],
            name = f'Start O:{occurence}',
            opacity = 0.7,
            showlegend = False,
            marker = dict(color=colors[0], size=3)
            ))
        
        fig.add_trace(go.Scatter3d(
            x=[tmpdf['speed'].iloc[-1]],
            y=[tmpdf['bl'].iloc[-1]],
            z=[tmpdf['yaw'].iloc[-1]],
            name = f'End O:{occurence}',
            opacity = 0.7,
            showlegend = False,
            marker = dict(color=colors[-1], size=3)
            ))
    """
    for xyz in ['x','y','z']:
        fig['layout'][f'scene{i+1}'][f'{xyz}axis']['backgroundcolor'] = 'rgb(200,200,230)'
        fig['layout'][f'scene{i+1}'][f'{xyz}axis']['range'] = [ranges[f'{xyz}min'],ranges[f'{xyz}max']]
        fig['layout'][f'scene{i+1}'][f'{xyz}axis']['title'] = titles[xyz]
        fig['layout'][f'scene{i+1}'][f'{xyz}axis']['nticks'] = 8
    """


    fig.update_layout(height=1000,width=1000)
    """   
    annotations = tools.make_subplots(rows=1, cols=1,
    subplot_titles=[f'State {k} , {max_occ[i]} trajectories' for i,k in enumerate(values)]
            )['layout']['annotations']
    """
    #fig['layout'].update(annotations=annotations)
    savename = f'example_state_{k}.html'

    fig.write_html(savename)
    print(k)