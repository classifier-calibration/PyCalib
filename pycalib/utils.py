import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def multiindex_to_strings(index):
    if isinstance(index, pd.MultiIndex):
        return [' '.join(col).strip() for col in index.values]
    return [''.join(col).strip() for col in index.values]


def df_normalise(df, columns=True):
    '''
    rows: bool
        Normalize each column to sum to one, or each row to sum to one
    '''
    if columns:
        return df/df.sum(axis=0)
    return (df.T/df.sum(axis=1)).T


def plot_df_to_heatmap(df, title=None, figsize=None, annotate=True,
                 normalise_columns=False, normalise_rows=False, cmap=None):
    ''' Exports a heatmap of the given pandas DataFrame

    Parameters
    ----------
    df:     pandas.DataFrame
        It should be a matrix, it can have multiple index and these will be
        flattened.

    title: string
        Title of the figure

    figsize:    tuple of ints (x, y)
        Figure size in inches

    annotate:   bool
        If true, adds numbers inside each box
    '''
    if normalise_columns:
        df = df_normalise(df, columns=True)
    if normalise_rows:
        df = df_normalise(df, columns=False)

    yticklabels = multiindex_to_strings(df.index)
    xticklabels = multiindex_to_strings(df.columns)
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        point_inch_ratio = 72.
        n_rows = df.shape[0]
        font_size_pt = plt.rcParams['font.size']
        xlabel_space_pt = max([len(xlabel) for xlabel in xticklabels])
        fig_height_in = ((xlabel_space_pt + n_rows) * (font_size_pt + 3)) / point_inch_ratio

        n_cols = df.shape[1]
        fig_width_in = df.shape[1]+4
        ylabel_space_pt = max([len(ylabel) for ylabel in yticklabels])
        fig_width_in = ((ylabel_space_pt + (n_cols * 3) + 5)
                        * (font_size_pt + 3)) / point_inch_ratio
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    cax = ax.pcolor(df, cmap=cmap)
    fig.colorbar(cax)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    middle_value = (df.max().max() + df.min().min())/2.0
    if annotate:
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                color = 'white' if middle_value > df.values[y, x] else 'black'
                plt.text(x + 0.5, y + 0.5, '%.2f' % df.values[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=color
                         )
    return fig
