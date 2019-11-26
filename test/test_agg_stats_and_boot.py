import numpy as np
import bootstrapped.stats_functions as bs_stats
import pytest

from metcalcpy.agg_stat import AggStat, pd
from metcalcpy.bootstrap_custom import bootstrap_and_value

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
        if results.lower_bound <= 0 and results.upper_bound >= 0:
            reject[ind] = 0

    # get the number of rejected
    number_of_rejected = sum(x == 1 for x in reject)
    percent_of_rejected = number_of_rejected * 100 / TEST_LENGTH
    return percent_of_rejected


def test_prepare_sl1l2_data(settings):
    agg_stat = settings['agg_stat']
    series_data = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d01')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat.statistic = agg_stat.params['list_stat'][0].lower()
    agg_stat._prepare_sl1l2_data(series_data)
    result = np.array([6532.28230, 4034.79153, 7108.67791, 5379.89956])
    assert np.allclose(result, series_data['fbar'])


def test_calc_stats(settings):
    agg_stat = settings['agg_stat']
    series_data = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d01')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat.statistic = agg_stat.params['list_stat'][0].lower()
    agg_stat._prepare_sl1l2_data(series_data)
    stat_val = agg_stat._calc_stats(series_data.values)
    assert np.allclose(np.array([192.13042749999997]), stat_val)


def test_calc_stats_derived(settings):
    agg_stat = settings['agg_stat']
    series_data_1 = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d01')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat.statistic = agg_stat.params['list_stat'][0].lower()
    agg_stat._prepare_sl1l2_data(series_data_1)

    series_data_2 = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d02')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat._prepare_sl1l2_data(series_data_2)

    operation = np.full((len(series_data_1.values), 1), "DIFF")
    values_both_arrays = np.concatenate((series_data_1.values, series_data_2.values), axis=1)
    values_both_arrays = np.concatenate((values_both_arrays, operation), axis=1)
    stat_val = agg_stat._calc_stats_derived(values_both_arrays)

    assert np.allclose(np.array([0.0036222619046952786]), stat_val)


def test_get_derived_series(settings):
    agg_stat = settings['agg_stat']
    series_val = agg_stat.params['series_val']
    indy_vals = agg_stat.params['indy_vals']
    result = agg_stat._get_derived_series(series_val, indy_vals)
    expected = [('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '0', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '30000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '60000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '90000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '120000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '150000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '180000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '210000', 'FBAR'),
                ('DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)', '240000', 'FBAR')]
    assert result == expected


def test_calculate_value_and_ci(settings):
    agg_stat = settings['agg_stat']
    agg_stat.calculate_value_and_ci()
    result_frame = pd.read_csv(
        agg_stat.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.size == 216
    assert result_frame.shape == (27, 8)
    assert result_frame['stat_value'][2] == 192.1304275
    assert result_frame['stat_value'][20] == 0.003622261904695279
    assert result_frame['stat_bcl'][9] == 192.12478000000002
    assert result_frame['stat_bcu'][24] == 0.010786756756743898


@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    params = {'random_seed': 1, 'indy_var': 'fcst_lead', 'list_static_val': {'fcst_var': 'DPT'}, 'method': 'perc',
              'num_iterations': 100, 'event_equal': 'True',
              'derived_series': [['ENS001v3.6.1_d01 DPT FBAR', 'ENS001v3.6.1_d02 DPT FBAR', 'DIFF']],
              'agg_stat_input': 'data/agg_stat_and_boot_data.data',
              'fcst_var_val': {'DPT': ['FBAR']},
              'agg_stat_output': 'data/agg_stat_and_boot_output.data',
              'fixed_vars_vals_input': {'fcst_lev': {'fcst_lev_0': ['P100']}},
              'series_val': {'model': ['ENS001v3.6.1_d01', 'ENS001v3.6.1_d02']}, 'alpha': 0.05, 'line_type': 'sl1l2',
              'num_threads': -1,
              'indy_vals': ['0', '30000', '60000', '90000', '120000', '150000', '180000', '210000', '240000'],
              'list_stat': ['FBAR']}
    agg_stat = AggStat(params)
    settings_dict = dict()
    settings_dict['agg_stat'] = agg_stat
    return settings_dict
