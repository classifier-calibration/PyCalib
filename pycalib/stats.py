from collections import namedtuple
import numpy as np
import pandas as pd

from functools import partial
from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare


TestResult = namedtuple("TestResult", ["statistic", "p_value"])


def compute_friedmanchisquare(table: pd.DataFrame) -> TestResult:
    """ Compute Friedman test for repeated samples
    Example:
        - n wine judges each rate k different wines. Are any of the k wines
        ranked consistently higher or lower than the others?
    Our Calibration case:
        - n datasets each rate k different calibration methods. Are any of the
        k calibration methods ranked consistently higher or lower than the
        others?
    This will output a statistic and a p-value
    SciPy does the following:
        - k: is the number of parameters passed to the function
        - n: is the length of each array passed to the function
    The two options for the given table are:
        - k is the datasets: table['mean'].values.tolist()
        - k is the calibration methods: table['mean'].T.values.tolist()
    """
    if table.shape[1] < 3:
        print('Friedman test not appropriate for less than 3 methods')
        return TestResult(np.nan, np.nan)

    statistic, p = friedmanchisquare(*table.T.values)
    return TestResult(statistic, p)


def paired_test(table, stats_func=ranksums):
    measure = table.columns.levels[0].values[0]
    pvalues = np.zeros((table.columns.shape[0], table.columns.shape[0]))
    statistics = np.zeros_like(pvalues)
    for i, method_i in enumerate(table.columns.levels[1]):
        for j, method_j in enumerate(table.columns.levels[1]):
            sample_i = table[measure, method_i]
            sample_j = table[measure, method_j]
            statistic, pvalue = stats_func(sample_i, sample_j)
            pvalues[i, j] = pvalue
            statistics[i, j] = statistic
    index = pd.MultiIndex.from_product([table.columns.levels[1],
                                        ['statistic']])
    df_statistics = pd.DataFrame(statistics,
                                 index=table.columns.levels[1],
                                 columns=index)
    index = pd.MultiIndex.from_product([table.columns.levels[1],
                                        ['pvalue']])
    df_pvalues = pd.DataFrame(pvalues,
                              index=table.columns.levels[1],
                              columns=index)
    return df_statistics.join(df_pvalues)


def compute_ranksums(table):
    return paired_test(table, stats_func=ranksums)


def compute_mannwhitneyu(table):
    return paired_test(table, stats_func=partial(mannwhitneyu,
                                                 alternative='less'))
