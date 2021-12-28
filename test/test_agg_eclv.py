import numpy as np
import pytest

from metcalcpy.agg_eclv import AggEclv, pd


def test_calculate_value_and_ci(settings):
    agg_eclv = settings['agg_stat']
    agg_eclv.calculate_stats_and_ci()
    result_frame = pd.read_csv(
        agg_eclv.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.size == 304
    assert result_frame.shape == (38, 8)
    assert np.allclose(result_frame['y_pnt_i'][2], 0.5285295)
    assert np.allclose(result_frame['y_pnt_i'][26], 0.65747)
    assert np.allclose(result_frame['nstats'][0], 23)
    assert result_frame['stat_btcl'][9] <= result_frame['y_pnt_i'][9] <= result_frame['stat_btcu'][9]
    assert result_frame['stat_btcl'][24] <= result_frame['y_pnt_i'][24] <= result_frame['stat_btcu'][24]


@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    params = {'random_seed': 1, 'indy_var': '',
              'method': 'perc',
              'num_iterations': 100, 'event_equal': 'False',
              'agg_stat_input': 'data/agg_eclv_data.data',
              'agg_stat_output': 'data/agg_eclv_data_output.data',
              'fixed_vars_vals_input': {},
              'series_val_1': {'model': ['WRF'], 'fcst_lev': ['Z10', 'P850-700']},
              'alpha': 0.05, 'line_type': 'ctc',
              'num_threads': -1,
              'indy_vals': [],
              'agg_stat1': ['ECLV'],
              'circular_block_bootstrap': True,
              'equalize_by_indep': 'True',
              'cl_step': 0.05
              }
    agg_stat = AggEclv(params)
    settings_dict = dict()
    settings_dict['agg_stat'] = agg_stat
    return settings_dict
