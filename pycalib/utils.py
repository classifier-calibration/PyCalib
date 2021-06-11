import pandas as pd
import numpy as np


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



def get_binned_scores(labels, scores, bins=10):
    '''
    Parameters
    ==========
    labels : array (n_samples, )
        Labels indicating the true class.
    scores : matrix (n_samples, )
        Output probability scores for one or several methods.
    bins : int or list of floats
        Number of bins to create in the scores' space, or list of bin
        boundaries.
    '''
    if isinstance(bins, int):
        n_bins = bins
        bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    elif isinstance(bins, list) or isinstance(bins, np.ndarray):
        n_bins = len(bins) - 1
        bins = np.array(bins)
        if bins[0] == 0.0:
            bins[0] = 0 - 1e-8
        if bins[-1] == 1.0:
            bins[-1] = 1 + 1e-8

    scores = np.clip(scores, a_min=0, a_max=1)

    bin_idx = np.digitize(scores, bins) - 1

    bin_true = np.bincount(bin_idx, weights=labels,
                           minlength=n_bins)
    bin_pred = np.bincount(bin_idx, weights=scores,
                           minlength=n_bins)
    bin_total = np.bincount(bin_idx, minlength=n_bins)

    zero_idx = bin_total == 0
    avg_true = np.empty(bin_total.shape[0])
    avg_true.fill(np.nan)
    avg_true[~zero_idx] = np.divide(bin_true[~zero_idx],
                                    bin_total[~zero_idx])
    avg_pred = np.empty(bin_total.shape[0])
    avg_pred.fill(np.nan)
    avg_pred[~zero_idx] = np.divide(bin_pred[~zero_idx],
                                    bin_total[~zero_idx])
    return avg_true, avg_pred, bin_true, bin_total
