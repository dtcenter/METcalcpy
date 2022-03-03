# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: bootstrap.py
"""

import numpy as _np
from collections import Iterable
import multiprocessing as _multiprocessing
import scipy.sparse as _sparse

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


class BootstrapResults(object):
    def __init__(self, lower_bound, value, upper_bound):
        try:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.value = value
            if self.lower_bound > self.upper_bound:
                self.lower_bound, self.upper_bound = self.upper_bound, self.lower_bound
        except TypeError:
            pass
        self.values = None
        self.distributions = None

    def __str__(self):
        return '{1}    ({0}, {2})'.format(self.lower_bound, self.value,
                                          self.upper_bound)

    def __repr__(self):
        return self.__str__()

    def _apply(self, other, func):
        return BootstrapResults(func(self.lower_bound, other),
                                func(self.value, other),
                                func(self.upper_bound, other))

    def __add__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __radd__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __sub__(self, other):
        return self._apply(float(other), lambda x, other: x - other)

    def __rsub__(self, other):
        return self._apply(float(other), lambda x, other: other - x)

    def __mul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def __rmul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def error_width(self):
        '''Returns: upper_bound - lower_bound'''
        return self.upper_bound - self.lower_bound

    def error_fraction(self):
        '''Returns the error_width / value'''
        if self.value == 0:
            return _np.inf
        else:
            return self.error_width() / self.value

    def is_significant(self):
        return _np.sign(self.upper_bound) == _np.sign(self.lower_bound)

    def get_result(self):
        '''Returns:
            -1 if statistically significantly negative
            +1 if statistically significantly positive
            0 otherwise
        '''
        return int(self.is_significant()) * _np.sign(self.value)

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
                        save_data=True, save_distributions=False, block_length: int = 1, eclv: bool = False):
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
             eclv: indicates if this bootstrap estimate is for the Economic Cost Loss Relative Value or not
                Default (eclv = false)
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
    if eclv:
        result = _get_confidence_interval_and_value_eclv(bootstrap_dist, stat_val, alpha, ci_method)
    else:
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
    return BootstrapResults(lower_bound=low,
                            value=val,
                            upper_bound=high)


def _get_confidence_interval_and_value_eclv(bootstrap_dist, stat_val, alpha, ci_method):
    """Get the bootstrap confidence interval for a given distribution for the Economic Cost Loss Relative Value
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
        val = None
        low = None
        high = None

    else:
        bd = bootstrap_dist[bootstrap_dist != _np.array([None])]
        all_values = []
        for dist_member in bd:
            all_values.append(dist_member['V'].tolist())

        all_values_np = _np.array(all_values)
        steps_len = len(stat_val['cl'])
        none_in_values = len(stat_val['V']) != sum(x is not None for x in stat_val['V'])
        stat_btcl = [None] * steps_len
        stat_btcu = [None] * steps_len

        for ind in range(steps_len):
            low = None
            high = None
            column = all_values_np[:, ind]
            if ci_method == 'pivotal':
                low = 2 * stat_val - _np.percentile(column, 100 * (1 - alpha / 2.))
                high = 2 * stat_val - _np.percentile(column, 100 * (alpha / 2.))

            elif ci_method == 'perc':
                if _all_the_same(column):
                    print(f'All values of t are equal to {column[0]}. Cannot calculate confidence intervals')
                    low = None
                    high = None
                else:
                    if none_in_values:
                        low = _np.percentile(column, 100 * (alpha / 2.), interpolation='linear')
                        high = _np.percentile(column, 100 * (1 - alpha / 2.), interpolation='linear')
            stat_btcl[ind] = low
            stat_btcu[ind] = high

        val = stat_val
        low = stat_btcl
        high = stat_btcu

    return BootstrapResults(lower_bound=low,
                            value=val,
                            upper_bound=high)


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, float):
            for x in flatten(item):
                yield x
        else:
            yield item


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


def _validate_arrays(values_lists):
    t = values_lists[0]
    t_type = type(t)
    if not isinstance(t, _sparse.csr_matrix) and not isinstance(t, _np.ndarray):
        raise ValueError(('The arrays must either be of type '
                          'scipy.sparse.csr_matrix or numpy.array'))

    for _, values in enumerate(values_lists[1:]):
        if not isinstance(values, t_type):
            raise ValueError('The arrays must all be of the same type')

        if t.shape != values.shape:
            raise ValueError('The arrays must all be of the same shape')

        if isinstance(t, _sparse.csr_matrix):
            if values.shape[0] > 1:
                raise ValueError(('The sparse matrix must have shape 1 row X N'
                                  ' columns'))

    if isinstance(t, _sparse.csr_matrix):
        if _needs_sparse_unification(values_lists):
            raise ValueError(('The non-zero entries in the sparse arrays'
                              ' must be aligned'))


def _needs_sparse_unification(values_lists):
    non_zeros = values_lists[0] != 0

    for v in values_lists:
        v_nz = v != 0
        non_zeros = (non_zeros + v_nz) > 0

    non_zero_size = non_zeros.sum()

    for v in values_lists:
        if non_zero_size != v.data.shape[0]:
            return True

    return False


def mean(values, axis=1):
    """Returns the mean of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.mean(axis=axis)
        return ret.A1
    else:
        return _np.mean(_np.asmatrix(values), axis=axis).A1


def sum(values, axis=1):
    """Returns the sum of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.sum(axis=axis)
        return ret.A1
    else:
        return _np.sum(_np.asmatrix(values), axis=axis).A1


def median(values, axis=1):
    """Returns the sum of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.median(axis=axis)
        return ret.A1
    else:
        return _np.median(_np.asmatrix(values), axis=axis).A1


def std(values, axis=1):
    """ Returns the std of each row of a matrix"""
    if isinstance(values, _sparse.csr_matrix):
        ret = values.std(axis=axis)
        return ret.A1
    else:
        return _np.std(_np.asmatrix(values), axis=axis).A1
