import numpy as np
import pytest
import math
import statistics

from metcalcpy import bootstrap
from metcalcpy.agg_stat import AggStat, pd
from metcalcpy.bootstrap import bootstrap_and_value
from metcalcpy.util.utils import round_half_up, PRECISION

TEST_LENGTH = 1000


def lossdiff_ml(data, ):
    if len(data.shape) < 3:
        lossdiff = data[:, 0] - data[:, 1]
        return [statistics.mean(lossdiff)]
    else:
        result = []
        for i in range(0, data.shape[0]):
            lossdiff = data[i][:, 0] - data[i][:, 1]
            result.append(statistics.mean(lossdiff))
        return result


def lossdiff_mal(data):
    if len(data.shape) < 3:
        ALlossdiff = abs(data[:, 0]) - abs(data[:, 1])
        return [statistics.mean(ALlossdiff)]
    else:
        result = []
        for i in range(0, data.shape[0]):
            ALlossdiff = abs(data[i][:, 0]) - abs(data[i][:, 1])
            result.append(statistics.mean(ALlossdiff))
        return result


def lossdiff_msl(data):
    if len(data.shape) < 3:
        SLlossdiff = data[:, 0] * data[:, 0] - data[:, 1] * data[:, 1]
        return [statistics.mean(SLlossdiff)]
    else:
        result = []
        for i in range(0, data.shape[0]):
            SLlossdiff = data[i][:, 0] * data[i][:, 0] - data[i][:, 1] * data[i][:, 1]
            result.append(statistics.mean(SLlossdiff))
        return result

@pytest.mark.skip("Not to be run in regression testing, due to extensive "
                  "number of data points.  This test takes a long time to run.")
def test_cboot():

    et = np.loadtxt(
        "./data/et.txt" )

    # create an array for accepted/rejected flags
    ml_reject = [1] * TEST_LENGTH
    mal_reject = [1] * TEST_LENGTH
    msl_reject = [1] * TEST_LENGTH
    mean_reject = [1] * TEST_LENGTH
    # run the boot ci TEST_LENGTH times
    for ind in range(TEST_LENGTH):
        results_ml = bootstrap_and_value(
            et,
            stat_func=lossdiff_ml,
            num_iterations=500, alpha=0.05,
            num_threads=1, ci_method='perc', block_length=32)

        # record if 0 in ci bounds (accept) = 0 or not (reject) = 1
        if results_ml.lower_bound <= 0 and results_ml.upper_bound >= 0:
            ml_reject[ind] = 0

        results_mal = bootstrap_and_value(
            et,
            stat_func=lossdiff_mal,
            num_iterations=500, alpha=0.05,
            num_threads=1, ci_method='perc', block_length=32)
        # record if 0 in ci bounds (accept)=0 or not (reject)=1
        if results_mal.lower_bound <= 0 and results_mal.upper_bound >= 0:
            mal_reject[ind] = 0

        results_msl = bootstrap_and_value(
            et,
            stat_func=lossdiff_msl,
            num_iterations=500, alpha=0.05,
            num_threads=1, ci_method='perc', block_length=32)

        # record if 0 in ci bounds (accept)=0 or not (reject)=1
        if results_msl.lower_bound <= 0 and results_msl.upper_bound >= 0:
            msl_reject[ind] = 0

        results_mean = bootstrap_and_value(
            et[:, 0],
            stat_func=bootstrap.mean,
            num_iterations=500, alpha=0.05,
            num_threads=1, ci_method='perc', block_length=32)
        if results_mean.lower_bound <= 0 and results_mean.upper_bound >= 0:
            mean_reject[ind] = 0

    # get the number of rejected
    ml_number_of_rejected = sum(x == 1 for x in ml_reject)
    ml_percent_of_rejected = ml_number_of_rejected * 100 / TEST_LENGTH
    ml_frequencies_of_ml_rejected = ml_number_of_rejected / TEST_LENGTH

    msl_number_of_rejected = sum(x == 1 for x in msl_reject)
    msl_percent_of_rejected = msl_number_of_rejected * 100 / TEST_LENGTH
    msl_frequencies_of_rejected = msl_number_of_rejected / TEST_LENGTH

    mal_number_of_rejected = sum(x == 1 for x in mal_reject)
    mal_percent_of_rejected = mal_number_of_rejected * 100 / TEST_LENGTH
    mal_frequencies_of_rejected = mal_number_of_rejected / TEST_LENGTH

    mean_number_of_rejected = sum(x == 1 for x in mean_reject)
    mean_percent_of_rejected = mean_number_of_rejected * 100 / TEST_LENGTH
    mean_frequencies_of_rejected = mean_number_of_rejected / TEST_LENGTH

    print('for ML   p = {} total rejected = {} frequency = {}'.format(ml_percent_of_rejected, ml_percent_of_rejected,
                                                                      ml_frequencies_of_ml_rejected))
    print('for MAL  p = {} total rejected = {} frequency = {}'.format(mal_percent_of_rejected, mal_number_of_rejected,
                                                                      mal_frequencies_of_rejected))
    print('for MSL  p = {} total rejected = {} frequency = {}'.format(msl_percent_of_rejected, msl_number_of_rejected,
                                                                      msl_frequencies_of_rejected))
    print(
        'for mean  p = {} total rejected = {} frequency = {}'.format(mean_percent_of_rejected, mean_number_of_rejected,
                                                                     mean_frequencies_of_rejected))

@pytest.mark.skip("Not to be run in as a regression test, it uses an extensive number of points"
                  " and takes a long time (well beyond 5 minutes) to run.")
def test_boot():
    # size of array
    n = 100
    for mean in range(6):
        p = get_rejected(mean, n)
        print('for mean = {} p = {}'.format(mean, p))


def get_rejected(mean, n):
    """Calculate the percent of rejected values for 0 hypothesis test
        for CI for mean statistic of the normal distribution of 100 values

        Args:
            mean - mean value for the normal distribution
            n - size of array
        Returns:
            percent of rejected values
    """

    block_lenght = int(math.sqrt(n))

    # create an array for accepted/rejected flags
    reject = [1] * TEST_LENGTH
    # run the boot ci TEST_LENGTH times
    for ind in range(TEST_LENGTH):
        # create normal distribution
        data = np.random.normal(loc=mean, size=n, scale=10)
        # get ci for mean stat for this distribution
        results = bootstrap_and_value(
            data,
            stat_func=bootstrap.mean,
            num_iterations=500, alpha=0.05,
            num_threads=1, ci_method='perc', block_length=block_lenght)

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
    agg_stat.statistic = agg_stat.params['list_stat_1'][0].lower()
    agg_stat._prepare_sl1l2_data(series_data)
    result = np.array([6532.28230, 4034.79153, 7108.67791, 5379.89956])
    assert np.allclose(result, series_data['fbar'])


def test_calc_stats(settings):
    agg_stat = settings['agg_stat']
    series_data = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d01')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat.statistic = agg_stat.params['list_stat_1'][0].lower()
    agg_stat._prepare_sl1l2_data(series_data)
    stat_val = agg_stat._calc_stats(series_data.values)
    assert np.allclose(np.array([192.13042749999997]), stat_val)


def test_calc_stats_derived(settings):
    agg_stat = settings['agg_stat']
    series_data_1 = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d01')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat.statistic = agg_stat.params['list_stat_1'][0].lower()
    agg_stat._prepare_sl1l2_data(series_data_1)

    series_data_2 = agg_stat.input_data[
        (agg_stat.input_data['model'] == 'ENS001v3.6.1_d02')
        & (agg_stat.input_data["fcst_lead"] == 60000)]
    agg_stat._prepare_sl1l2_data(series_data_2)

    operation = np.full((len(series_data_1.values), 1), "DIFF")
    values_both_arrays = np.concatenate((series_data_1.values, series_data_2.values), axis=1)
    values_both_arrays = np.concatenate((values_both_arrays, operation), axis=1)
    stat_val = agg_stat._calc_stats_derived(values_both_arrays)

    assert np.allclose(np.array([0.00362229]), stat_val)


def test_get_derived_series(settings):
    agg_stat = settings['agg_stat']
    series_val = agg_stat.params['series_val_1']
    indy_vals = agg_stat.params['indy_vals']
    result = agg_stat._get_derived_points(series_val, indy_vals)
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
    agg_stat.calculate_stats_and_ci()
    result_frame = pd.read_csv(
        agg_stat.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.size == 216
    assert result_frame.shape == (27, 8)
    assert np.allclose(result_frame['stat_value'][2], 192.1304275)
    assert np.allclose(result_frame['stat_value'][20], 0.00362229)
    assert result_frame['stat_btcl'][9] <= result_frame['stat_value'][9] <=result_frame['stat_btcu'][9]
    assert result_frame['stat_btcl'][24] <= result_frame['stat_value'][24] <=result_frame['stat_btcu'][24]


@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    params = {'random_seed': 1, 'indy_var': 'fcst_lead',
               'method': 'perc',
              'num_iterations': 100, 'event_equal': 'True',
              'derived_series_1': [
                  ['ENS001v3.6.1_d01 DPT FBAR', 'ENS001v3.6.1_d02 DPT FBAR', 'DIFF']],
              'derived_series_2': [],
              'agg_stat_input': 'data/agg_stat_and_boot_data.data',
              'fcst_var_val_1': {'DPT': ['FBAR']},
              'fcst_var_val_2': {},
              'agg_stat_output': 'data/agg_stat_and_boot_output.data',
              'fixed_vars_vals_input': {'fcst_lev': {'fcst_lev_0': ['P100']}},
              'series_val_1': {'model': ['ENS001v3.6.1_d01', 'ENS001v3.6.1_d02']},
              'series_val_2': {},
              'alpha': 0.05, 'line_type': 'sl1l2',
              'num_threads': -1,
              'indy_vals': ['0', '30000', '60000', '90000',
                            '120000', '150000', '180000', '210000', '240000'],
              'list_stat_1': ['FBAR'],
              'list_stat_2': [],
              'circular_block_bootstrap': False}
    agg_stat = AggStat(params)
    settings_dict = dict()
    settings_dict['agg_stat'] = agg_stat
    return settings_dict
