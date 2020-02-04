import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_colored_seasonal_decompose(timeseries, colors=('k', 'y', 'm', 'b'), **kwargs):
    fig = seasonal_decompose(timeseries, **kwargs).plot()
    for ax, line_color in zip(fig.axes, colors):
        ax.get_lines()[0].set_color(line_color)
    return fig

def timeindex_to_yearfraction(timeindex, t0=None):
    t0 = t0 or timeindex[0]
    seconds_since_start = (timeindex - t0).total_seconds()
    seconds_in_avg_year = (60*60*24*365.25)               
    return  (seconds_since_start/seconds_in_avg_year).values

def scatter_with_labels(df, x, y, val):
    ax = df.plot.scatter(x=x, y=y)
    for _, row in df.iterrows():
        ax.text(row[x], row[y], str(row[val]))
        
def name_params(model):
    ar_param_labels = [f'ar{i+1}' for i,_ in enumerate(model.arparams())]
    ma_param_labels = [f'ma{i+1}' for i,_ in enumerate(model.maparams())]
    params = pd.Series(
        model.params().copy(), 
        index=['const', 't',  'sin', 'cos'] + ar_param_labels + ma_param_labels
    )
    return params
        
def plot_trend_and_seasonal_effects(t, values, params, figsize=(20,4)):
    trend_effect = params['const'] + t*params['t']
    seasonal_effect = params['sin']*np.sin(2*np.pi*t) + params['cos']*np.cos(2*np.pi*t) 
    
    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(*figsize)
    
    # Long-Term Trend
    y_points = values - seasonal_effect
    axes[0].plot(t, y_points, 'yo', alpha=0.6)
    axes[0].plot(t, trend_effect, 'm')
    axes[0].set_title("Long-term trend", fontsize=18)
    
    # Seasonal Effect
    t_mod = (t.round(4) % 1)
    t_line = np.linspace(0, 1, 100)
    y_points = values - trend_effect
    y_line = params['sin']*np.sin(2*np.pi*t_line) + params['cos']*np.cos(2*np.pi*t_line) 
    axes[1].plot(t_mod, y_points, 'yo', alpha=0.6)
    axes[1].plot(t_line, y_line, 'm')
    axes[1].set_title("Seasonal Effect", fontsize=18)
    
    return fig