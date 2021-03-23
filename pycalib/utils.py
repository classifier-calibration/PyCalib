import pandas as pd


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
