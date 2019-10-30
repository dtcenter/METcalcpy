from metcalcpy.bootstrap_custom import bootstrap_and_value
import numpy as np
import bootstrapped.stats_functions as bs_stats

TEST_LENGTH = 1000


def test_boot():
    for mean in range(6):
        p = get_rejected(mean)
        print('for mean = {} p = {}'.format(mean, p))


def get_rejected(mean):
    """Calculate the percent of rejected values for 0 hypothesis test
        for CI for mean statistic of the normal distribution of 100 values

        Args:
            mean - mean value for the normal distribution
        Returns:
            percent of rejected values
    """

    # create an array for accepted/rejected flags
    reject = [1] * TEST_LENGTH
    # run the boot ci TEST_LENGTH times
    for ind in range(TEST_LENGTH):
        # create normal distribution
        data = np.random.normal(loc=mean, size=100, scale=10)
        # get ci for mean stat for this distribution
        results = bootstrap_and_value(
            data,
            stat_func=bs_stats.mean,
            num_iterations=1000, alpha=0.05,
            num_threads=1, ci_method='perc')

        # record if 0 in ci bounds (accept) or not (reject)
        if 0 >= results.lower_bound and 0 <= results.upper_bound:
            reject[ind] = 0

    # get the number of rejected
    number_of_rejected = sum(x == 1 for x in reject)
    percent_of_rejected = number_of_rejected * 100 / TEST_LENGTH
    return percent_of_rejected


if __name__ == "__main__":
    test_boot()