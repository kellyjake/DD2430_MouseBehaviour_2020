import pandas as pd
import numpy as np
import seaborn as sns
from ssm.plots import gradient_cmap
import plotly.graph_objects as go
from IPython.display import HTML , display

df = pd.DataFrame(data={'item':list(range(20)),
                        'poop':list(np.cos(range(20))),
                        'whatev':list(np.sin(range(20)))})

df['state_consec'] = (df.item.diff(1) != 0).astype('int')
df['test'] = df.groupby('item')['state_consec'].cumsum()

print(df[::10])

def rgb_to_hex(rgb):    
    return f'#{int(rgb[0]*256):02x}{int(rgb[1]*256):02x}{int(rgb[2]*256):02x}'

print(np.unique(df['test']))

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

max_appearence = max(df['test'])
values = np.unique(df['test'])
colors = [ cmap(value/max_appearence)[:-1] for value in values ]

cols = list(map(lambda x : rgb_to_hex(colors[x-1]), df['test']))


df['u'] = df.item.diff(1)
df['v'] = df.poop.diff(1)
df['w'] = df.whatev.diff(1)

df['x'] = df.item.rolling(2).sum() / 2
df['y'] = df.poop.rolling(2).sum() / 2
df['z'] = df.whatev.rolling(2).sum() / 2

df['norm'] = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)

df['u_normed'] = df['u'] / df['norm']
df['v_normed'] = df['v'] / df['norm']
df['w_normed'] = df['w'] / df['norm']

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=df['item'],
    y=df['poop'],
    z=df['whatev']
))

fig.add_trace(go.Cone(  x=df['x'][1:],
                        y=df['y'][1:],
                        z=df['z'][1:],
                        u=df['u_normed'][1:],
                        v=df['v_normed'][1:],
                        w=df['w_normed'][1:],
                        showscale=False,
                        opacity=0.5))

fig.update_layout(scene=dict(aspectmode='cube'))

fig.write_html('test.html')
#display(HTML(fig.to_html()))