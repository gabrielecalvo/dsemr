import pandas as pd
import matplotlib.pyplot as plt


def format_ax(ax, xlab, ylab, title):
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    return ax

    
def reliability(df, actual, ax=None):
    res = {}
    for col in df:
        alpha = int(col[1:])/100
        res[alpha] = (actual < df[col]).mean()
    res = pd.Series(res)
    
    if ax:
        res.plot(style='o-', grid=True,ax=ax)
        ax.plot([0,1], [0,1], 'k--')
        format_ax(ax, xlab='Model', ylab='Actual', title='Reliability')
        
    return res

def sharpness(df, actual, ax=None):
    model_res = {}
    actual_res = {}
    for halfalpha in range(5, 46, 5):
        x_label = halfalpha*2/100
        
        upper_col = f'q{50+halfalpha}'
        lower_col = f'q{50-halfalpha}'
        model_res[x_label] = (df[upper_col]-df[lower_col]).mean()
        
        upper = actual.quantile((50+halfalpha)/100)
        lower = actual.quantile((50-halfalpha)/100)
        actual_res[x_label] = upper-lower
        
    model_res = pd.Series(model_res)
    actual_res = pd.Series(actual_res)
    
    if ax:
        model_res.plot(style='o-', grid=True, ax=ax)
        actual_res.plot(style='y-.', grid=True, alpha=0.2, ax=ax)
        format_ax(ax, xlab='Alpha', ylab='Interval Width', title='Sharpness')
        
    return model_res

def eval_plots(df, actual, figsize=(20,8), title=''):
    fig, axes = plt.subplots(1,2)
    fig.suptitle(title)
    fig.set_size_inches(*figsize)
    reliability(df, actual, ax=axes[0])
    sharpness(df, actual, ax=axes[1])
    return fig


def pinball_loss(y, z, tau):
    loss = np.where(y>=z, (y-z)*tau, (z-y)*(1-tau))
    return np.nansum(loss)