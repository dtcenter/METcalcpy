# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
from typing import Union
import math

from statistics import mean, median
import numpy as np
from itertools import groupby
from scipy.stats import norm

from metcalcpy.util.correlation import remove_none


def runs_test(x, alternative="two.sided", threshold='median') -> dict:
    """
    Wald-Wolfowitz Runs Test.
    Performs the Wald-Wolfowitz runs test of randomness for continuous data.
    Method used to compute the p-value is normal
    Mimics runs.test Rscript function

    :param x: a numeric vector containing the observations
    :param alternative: a character string with the alternative hypothesis.
        Must be one of "two.sided" (default), "left.sided" or "right.sided".
    :param threshold: the cut-point to transform the data into a dichotomous vector
        Must be one of "median" (default) or "mean".
    :return: Wald-Wolfowitz Runs Test results as a dictionary
        statistic - the value of the normalized statistic test
        p_value - the p-value of the test
        runs - the total number of runs
        mean_value - the mean value of the statistic test
        variance - the variance of the statistic test
    """
    result = {
        'statistic': None,
        'p_value': None,
        'runs': None,
        'mean_value ': None,
        'variance': None
    }
    if alternative != "two.sided" and alternative != "left.sided" and alternative != "right.sided":
        print(" runs_test must get a valid alternative")
        return result
    if len(x) == 0:
        return result

    x = remove_none(x)
    if threshold == 'median':
        x_threshold = median(x)
    elif threshold == "mean":
        x_threshold = mean(x)
    else:
        print('ERROR  incorrect threshold')
        x_threshold = None
    x = [elem for elem in x if elem != x_threshold]
    res = [i - x_threshold for i in x]
    s = np.sign(res)
    n1 = 0
    n2 = 0
    for num in s:
        if num > 0:
            n1 += 1
        elif num < 0:
            n2 += 1
    runs = [(k, sum(1 for i in g)) for k, g in groupby(s)]
    r1 = 0
    r2 = 0
    for run in runs:
        if run[0] == 1:
            r1 += 1
        elif run[0] == -1:
            r2 += 1
    n = n1 + n2
    mean_value = 1 + 2 * n1 * n2 / (n1 + n2)
    variance = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (n * n * (n - 1))
    rr = r1 + r2
    pv = 0
    pv0 = norm.cdf((rr - mean_value) / math.sqrt(variance))
    if alternative == "two.sided":
        pv = 2 * min(pv0, 1 - pv0)
    if alternative == "left.sided":
        pv = pv0
    if alternative == "right.sided":
        pv = 1 - pv0

    result['statistic'] = (rr - mean_value) / math.sqrt(variance)
    result['p_value'] = pv
    result['runs'] = rr
    result['mean_value'] = mean_value
    result['variance'] = variance

    return result
