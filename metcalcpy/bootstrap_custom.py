"""
Program Name: bootstrap_custom.py
"""

import numpy as _np
import multiprocessing as _multiprocessing
import scipy.sparse as _sparse

from bootstrapped.bootstrap import BootstrapResults, _validate_arrays

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


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

    def set_distributions(self, distributions):
        """Sets distributions y
            Args: distributions - numpy array
        """
        self.distributions = distributions


def bootstrap_and_value(values, stat_func, alpha=0.05,
                        num_iterations=1000, iteration_batch_size=None,
                        num_threads=1, ci_method='perc',
                        save_data=True, save_distributions=False, block_length: int = 1):
    """Returns bootstrap estimate. Can do the independent and identically distributed (IID)
        or Circular Block Bootstrap (CBB) methods depending on the block_length
        Args:
            values: numpy array  of values to bootstrap
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
            save_data: Save or not the original data to the resulting object
            save_distributions: Save or not the distributions to the resulting object
            block_length: number giving the desired block lengths.
                Default (block.length = 1) is to do IID resamples.
                Should be longer than the length of dependence in the data,
                but much shorter than the size of the data. Generally, the square
                root of the sample size is a good choice
        Returns:
            BootstrapDistributionResults representing CI, stat value and the original distribution.
    """

    values_lists = [values]
    stat_func_lists = [stat_func]

    def do_division(distr):
        return distr

    stat_val = stat_func(values)[0]
    sz = num_iterations / block_length
    distributions = _bootstrap_distribution_cbb(values_lists,
                                                stat_func_lists,
                                                num_iterations,
                                                iteration_batch_size,
                                                num_threads, block_length)

    bootstrap_dist = do_division(*distributions)
    result = _get_confidence_interval_and_value(bootstrap_dist, stat_val, alpha, ci_method)
    if save_data:
        result.set_original_values(values)
    if save_distributions:
        result.set_distributions(bootstrap_dist.flatten('F'))
    return result


def _bootstrap_distribution_cbb(values_lists, stat_func_lists,
                                num_iterations, iteration_batch_size, num_threads, block_length=1):
    '''Returns the simulated bootstrap distribution. The idea is to sample the same
        indexes in a bootstrap re-sample across all arrays passed into values_lists.

        This is especially useful when you want to co-sample records in a ratio metric.
            numerator[k].sum() / denominator[k].sum()
        and not
            numerator[ j ].sum() / denominator[k].sum()
    Args:
        values_lists: list of numpy arrays
            each represents a set of values to bootstrap. All arrays in values_lists
            must be of the same length.
        stat_func_lists: statistic to bootstrap for each element in values_lists.
        num_iterations: number of bootstrap iterations / resamples / simulations
            to perform.
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
        block_length: number giving the desired block lengths.
                Default (block.length = 1) is to do IID resamples.
                Should be longer than the length of dependence in the data,
                but much shorter than the size of the data.Generally, the square
                root of the sample size is a good choice
    Returns:
        The set of bootstrap resamples where each stat_function is applied on
        the bootsrapped values.
    '''

    _validate_arrays(values_lists)

    if iteration_batch_size is None:
        iteration_batch_size = num_iterations

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    num_threads = int(num_threads)

    if num_threads == -1:
        num_threads = _multiprocessing.cpu_count()

    if num_threads <= 1:
        results = _bootstrap_sim_cbb(values_lists, stat_func_lists,
                                     num_iterations, iteration_batch_size, None, block_length)
    else:
        pool = _multiprocessing.Pool(num_threads)

        iter_per_job = _np.ceil(num_iterations * 1.0 / num_threads)

        results = []
        for seed in _np.random.randint(0, 2 ** 32 - 1, num_threads):
            r = pool.apply_async(_bootstrap_sim_cbb, (values_lists, stat_func_lists,
                                                      iter_per_job,
                                                      iteration_batch_size, seed, block_length))
            results.append(r)

        results = _np.hstack([res.get() for res in results])

        pool.close()

    return results


def bootstrap_and_value_mode(values, cases, stat_func, alpha=0.05,
                             num_iterations=1000, iteration_batch_size=None,
                             num_threads=1, ci_method='perc',
                             save_data=True, save_distributions=False, block_length=1):
    """Returns bootstrap estimate.
        Args:
            values: numpy array  of values to bootstrap
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
            save_data: Save or not the original data to the resulting object
            save_distributions: Save or not the distributions
            block_length: number giving the desired block lengths.
                Default (block.length = 1) is to do IID resamples.
                Should be longer than the length of dependence in the data,
                but much shorter than the size of the data. Generally, the square
                root of the sample size is a good choice
        Returns:
            BootstrapDistributionResults representing CI, stat value and the original distribution.
    """

    values_lists = [cases]

    stat_func_lists = [stat_func]

    def do_division(distr):
        return distr

    data_cases = _np.asarray(values['case'])
    flat_cases = cases.flatten()
    values_current = values[_np.in1d(data_cases, flat_cases)].to_numpy()
    stat_val = stat_func(values_current)[0]
    distributions = _bootstrap_distribution_cbb(values_lists,
                                                stat_func_lists,
                                                num_iterations,
                                                iteration_batch_size,
                                                num_threads, block_length)

    bootstrap_dist = do_division(*distributions)
    result = _get_confidence_interval_and_value(bootstrap_dist, stat_val, alpha, ci_method)
    if save_data:
        result.set_original_values(values)
    if save_distributions:
        result.set_distributions(bootstrap_dist.flatten('F'))
    return result


def _get_confidence_interval_and_value(bootstrap_dist, stat_val, alpha, ci_method):
    """Get the bootstrap confidence interval for a given distribution.
        Args:
            bootstrap_dist: numpy array of bootstrap results from
                bootstrap_distribution() or  bootstrap_distribution_cbb()
                or bootstrap_ab_distribution()
            stat_val: The overall statistic that this method is attempting to
                calculate error bars for.
            alpha: The alpha value for the confidence intervals.
            ci_method: if true, use the pivotal method. if false, use the
                percentile method.
    """

    # TODO Only percentile method for the confident intervals is implemented

    if stat_val is None:
        ci_method = "None"

    if ci_method == 'pivotal':
        low = 2 * stat_val - _np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
        val = stat_val
        high = 2 * stat_val - _np.percentile(bootstrap_dist, 100 * (alpha / 2.))
    elif ci_method == 'perc':
        # check if All values of bootstrap_dist are equal and if YES -
        # display a warning and do not calculate CIs - like boot.ci in R
        if _all_the_same(bootstrap_dist):
            print(f'All values of t are equal to {bootstrap_dist[0]}. Cannot calculate confidence intervals')
            low = None
            high = None
        else:
            bd = bootstrap_dist[bootstrap_dist != _np.array([None])]
            low = _np.percentile(bd, 100 * (alpha / 2.), interpolation='linear')
            high = _np.percentile(bd, 100 * (1 - alpha / 2.), interpolation='linear')
        val = stat_val
    else:
        low = None
        val = None
        high = None
    return BootstrapDistributionResults(lower_bound=low,
                                        value=val,
                                        upper_bound=high)


def _bootstrap_sim_cbb(values_lists, stat_func_lists, num_iterations,
                       iteration_batch_size, seed, block_length=1):
    """Returns simulated bootstrap distribution. Can do the independent and identically distributed (IID)
        or Circular Block Bootstrap (CBB) methods depending on the block_length
        Args:
            values_lists: numpy array  of values to bootstrap

            stat_func_lists: statistic to bootstrap

            num_iterations: number of bootstrap iterations to run. The higher this
            number the more sure you can be about the stability your bootstrap.

            iteration_batch_size: The bootstrap sample can generate very large
            matrices. This argument limits the memory footprint by
            batching bootstrap rounds. If unspecified the underlying code
            will produce a matrix of len(values) x num_iterations. If specified
            the code will produce sets of len(values) x iteration_batch_size
            (one at a time) until num_iterations have been simulated.

            seed: random seed

            block_length: number giving the desired block lengths.
                Default (block.length = 1) is to do IID resamples.
                Should be longer than the length of dependence in the data,
                but much shorter than the size of the data.
    """

    if seed is not None:
        _np.random.seed(seed)

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    results = [[] for _ in values_lists]

    for rng in range(0, num_iterations, iteration_batch_size):
        max_rng = min(iteration_batch_size, num_iterations - rng)

        values_sims = _generate_distributions_cbb(values_lists, max_rng, block_length)

        for i, values_sim, stat_func in zip(range(len(values_sims)), values_sims, stat_func_lists):
            results[i].extend(stat_func(values_sim))

    return _np.array(results)


def _generate_distributions_cbb(values_lists, num_iterations, block_length=1):
    values_shape = values_lists[0].shape[0]
    ids = _np.random.choice(
        values_shape,
        (num_iterations, values_shape),
        replace=True
    )

    def apply_cbb(row):
        """
        Applyes Circular Block Bootstrap (CBB) method to each row
        :param row:
        """
        counter = 0
        init_val = row[0]
        for ind, val in enumerate(row):
            if counter == 0:
                # save a 1st value for the block
                init_val = val
            else:
                # calculate current value by adding the counter to the initial value
                new_val = init_val + counter
                # the value should not be bigger then the size of the row
                if new_val > len(row) - 1:
                    new_val = new_val - len(row)
                row[ind] = new_val
            counter = counter + 1
            if counter == block_length:
                # start a new block
                counter = 0
        return row

    if block_length > 1:
        # uss CBB
        ids = _np.apply_along_axis(apply_cbb, axis=1, arr=ids)

    results = [values[ids] for values in values_lists]
    return results


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
    try:
        result = len(_np.unique(elements[elements != _np.array([None])])) == 1
    except TypeError:
        result = False
    return result
