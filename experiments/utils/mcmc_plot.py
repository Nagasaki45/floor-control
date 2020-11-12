import functools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def ax_plotter(f):
    '''
    A wrapper for functions that accept an ax and return an ax.
    '''
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
        f(*args, **kwargs)
        return kwargs['ax']
    return wrapper


@ax_plotter
def hdi(hdi_tails, ax):
    '''
    Plot a black line and HDI tails values at the
    bottom of the X axis.
    '''
    nodge = ax.get_ylim()[1] / 40

    for tail, alignment in zip(hdi_tails, ['right', 'left']):
        s = f'{tail:.2f}'.replace('-0', '-').lstrip('0')
        ax.text(tail, 2 * nodge, s, horizontalalignment=alignment)

    ax.plot(hdi_tails, [nodge, nodge], color='black', linewidth=2)


@ax_plotter
def dist(trace, ax, histplot_kwargs=None):
    '''
    Plot distribution analysis.
    '''
    if histplot_kwargs is None:
        histplot_kwargs = {}
    sns.histplot(trace, bins=20, ax=ax, **histplot_kwargs)
    ax.set(xlabel='', ylabel='', yticks=[])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    hdi_range = np.percentile(trace, [2.5, 97.5])
    hdi(hdi_range, ax=ax)


def param_comparison(trace, param, comparison, scatter_sample=30, names=None):
    '''
    Plot parameter comparison.
    '''
    n = len(comparison)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(8, 8))

    if names is None:
        names = {x: x for x in comparison}
    else:
        names = {x: y for x, y in zip(comparison, names)}

    x = trace[param]
    x = x.reshape((x.shape[0], -1))

    sample = np.random.randint(len(x[-1]), size=scatter_sample)

    for i, first in enumerate(comparison):
        dist(x[first], ax=axes[i, i])
        first_name = names[first]
        axes[i, i].set(title=f'{param}[{first_name}]')
        for j, second in enumerate(comparison[i + 1:], start=i + 1):
            dist(x[first] - x[second], ax=axes[i, j])
            axes[i, j].axvline(0, linestyle='--', c='grey')
            seconds_name = names[second]
            axes[i, j].set(title=f'{param}[{first_name}] - {param}[{seconds_name}]')
            axes[j, i].scatter(x[first][sample], x[second][sample], s=3)
            axes[j, i].set_adjustable('datalim')
            axes[j, i].set_aspect('equal')
            x_mean = x[first][sample].mean()
            axes[j, i].axline((x_mean, x_mean), slope=1, linestyle='--', color='grey')
            axes[j, i].set(xlabel=f'{param}[{first_name}]', ylabel=f'{param}[{seconds_name}]')

    fig.tight_layout()
