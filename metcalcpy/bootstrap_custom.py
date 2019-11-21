"""
Program Name: bootstrap_custom.py
"""

import numpy as _np
from bootstrapped.bootstrap import _bootstrap_distribution, BootstrapResults

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'

WARNING_MESSAGE = 'All values of t are equal to {}. Cannot calculate confidence intervals'


class BootstrapDistributionResults(BootstrapResults):
    """A class that extends BootstrapResults.
        Adds the original numpy array with the data for stat calculation
        as a 'values' variable
    """

    def __init__(self, lower_bound, value, upper_bound):
        try:
            super().__init__(lower_bound, value, upper_bound)
        except TypeError:
            pass
        self.values = None

    def set_original_values(self, values):
        """Sets values to the original data array
            Args: values - numpy array
        """
        self.values = values


def bootstrap_and_value(values, stat_func, alpha=0.05,
                        num_iterations=1000, iteration_batch_size=None,
                        num_threads=1, ci_method='perc'):
    """Returns bootstrap estimate.
        Args:
            values: numpy array (or scipy.sparse.csr_matrix) of values to bootstrap
            stat_func: statistic to bootstrap. We provide several default functions:
                    * stat_functions.mean
                    * stat_functions.sum
                    * stat_functions.std
            alpha: alpha value representing the confidence interval.
                Defaults to 0.05, i.e., 95th-CI.
            num_iterations: number of bootstrap iterations to run. The higher this
                number the more sure you can be about the stability your bootstrap.
                By this - we mean the returned interval should be consistent across
                runs for the same input. This also consumes more memory and makes
                analysis slower. Defaults to 10000.
            iteration_batch_size: The bootstrap sample can generate very large
                matrices. This argument limits the memory footprint by
                batching bootstrap rounds. If unspecified the underlying code
                will produce a matrix of len(values) x num_iterations. If specified
                the code will produce sets of len(values) x iteration_batch_size
                (one at a time) until num_iterations have been simulated.
                Defaults to no batching.

            num_threads: The number of therads to use. This speeds up calculation of
                the bootstrap. Defaults to 1. If -1 is specified then
                multiprocessing.cpu_count() is used instead.
            ci_method: method for bootstrapping confidence intervals.
        Returns:
            BootstrapDistributionResults representing CI, stat value and the original distribution.
    """

    values_lists = [values]
    stat_func_lists = [stat_func]

    def do_division(distr):
        return distr

    stat_val = stat_func(values)[0]
    distributions = _bootstrap_distribution(values_lists,
                                            stat_func_lists,
                                            num_iterations,
                                            iteration_batch_size,
                                            num_threads)

    bootstrap_dist = do_division(*distributions)
    result = _get_confidence_interval_and_value(bootstrap_dist, stat_val, alpha, ci_method)
    result.set_original_values(values)
    return result


def _get_confidence_interval_and_value(bootstrap_dist, stat_val, alpha, ci_method):
    """Get the bootstrap confidence interval for a given distribution.
        Args:
            bootstrap_dist: numpy array of bootstrap results from
                bootstrap_distribution() or bootstrap_ab_distribution()
            stat_val: The overall statistic that this method is attempting to
                calculate error bars for.
            alpha: The alpha value for the confidence intervals.
            ci_method: if true, use the pivotal method. if false, use the
                percentile method.
    """
    if ci_method == 'pivotal':
        low = 2 * stat_val - _np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
        val = stat_val
        high = 2 * stat_val - _np.percentile(bootstrap_dist, 100 * (alpha / 2.))
    elif ci_method == 'perc':
        # check if All values of bootstrap_dist are equal and if YES -
        # display a warning and do not calculate CIs - like boot.ci in R
        if _all_the_same(bootstrap_dist):
            print(WARNING_MESSAGE.format(bootstrap_dist[0]))
            low = None
            high = None
        else:
            low = _np.percentile(bootstrap_dist, 100 * (alpha / 2.), interpolation='linear')
            high = _np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.), interpolation='linear')
        val = stat_val
    else:
        low = None
        val = None
        high = None
    return BootstrapDistributionResults(lower_bound=low,
                                        value=val,
                                        upper_bound=high)


def _all_the_same(elements):
    """Checks if all elements in the numpy array are the same

         Args:
            elements: numpy array

        Returns:
            True if array is empty or all elements are the same
            False if the array has different elements
    """
    if elements.size == 0:
        return True
    return len(_np.unique(elements)) == 1
