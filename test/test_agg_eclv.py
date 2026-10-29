import os
import numpy as np
import pytest

from metcalcpy.agg_eclv import AggEclv, pd
from metcalcpy.bootstrap import _norm_inter, _boot_const

cwd = os.path.dirname(__file__)

def test_calculate_value_and_ci(settings):
    agg_eclv = settings['agg_stat']
    agg_eclv.calculate_stats_and_ci()
    result_frame = pd.read_csv(
        agg_eclv.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    # 19 cost/loss ratios plus the base rate (as in agg_eclv.R) for each of the 2 series
    assert result_frame.size == 320
    assert result_frame.shape == (40, 8)
    z10 = result_frame[result_frame['fcst_lev'] == 'Z10'].reset_index(drop=True)
    p850 = result_frame[result_frame['fcst_lev'] == 'P850-700'].reset_index(drop=True)
    assert np.allclose(z10[np.isclose(z10['x_pnt_i'], 0.15)]['y_pnt_i'], 0.5285295)
    assert np.allclose(p850[np.isclose(p850['x_pnt_i'], 0.4)]['y_pnt_i'], 0.65747)
    # the base rate is inserted in the sorted cost/loss ratios
    assert np.allclose(z10['x_pnt_i'][2], 0.144095)
    assert np.allclose(p850['x_pnt_i'][4], 0.217718)
    assert np.allclose(result_frame['nstats'][0], 23)
    assert result_frame['stat_btcl'][9] <= result_frame['y_pnt_i'][9] <= result_frame['stat_btcu'][9]
    assert result_frame['stat_btcl'][24] <= result_frame['y_pnt_i'][24] <= result_frame['stat_btcu'][24]


def test_calculate_value_and_ci_no_series(settings, tmp_path):
    '''
       With no series variables no filters are generated, so all rows
       must be used as a single point
    '''
    params = settings['agg_stat'].params
    params['series_val_1'] = {}
    params['agg_stat_output'] = f'{tmp_path}/agg_eclv_no_series_output.data'
    agg_eclv = AggEclv(params)
    agg_eclv.calculate_stats_and_ci()
    result_frame = pd.read_csv(
        agg_eclv.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.shape[0] == len(agg_eclv.steps) + 1
    assert np.allclose(result_frame['nstats'][0], len(agg_eclv.input_data))


def test_calculate_value_and_ci_group(settings, tmp_path):
    '''
       A group series value 'A:B' uses the rows of all members of the group, as in agg_eclv.R
    '''
    params = settings['agg_stat'].params
    params['series_val_1'] = {'model': ['WRF'], 'fcst_lev': ['Z10:P850-700']}
    params['agg_stat_output'] = f'{tmp_path}/agg_eclv_group_output.data'
    agg_eclv = AggEclv(params)
    agg_eclv.calculate_stats_and_ci()
    result_frame = pd.read_csv(
        agg_eclv.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.shape == (len(agg_eclv.steps) + 1, 8)
    assert (result_frame['fcst_lev'] == 'Z10:P850-700').all()
    assert np.allclose(result_frame['nstats'][0], 26)


def test_cl_steps():
    # expected values are from seq(clStep, 1-clStep, clStep) in R
    assert np.allclose(AggEclv._cl_steps(0.05), np.arange(1, 20) * 0.05)
    assert np.allclose(AggEclv._cl_steps(0.3), [0.3, 0.6])
    assert len(AggEclv._cl_steps(0.07)) == 13


def test_norm_inter():
    # expected values are from boot:::norm.inter in R
    t = np.arange(1, 101) ** 2 / 100
    assert np.allclose(_norm_inter(t, [0.025, 0.975]), [0.0683450869653095, 96.8932035735668], rtol=0, atol=1e-12)
    t = np.arange(1, 20) ** 2 / 10
    assert np.allclose(_norm_inter(t, [0.025, 0.975]), [0.1, 36.1])


def test_boot_const():
    assert _boot_const(np.array([0.2, 0.2, 0.2]))
    assert not _boot_const(np.array([0.2, 0.3, 0.2]))
    # boot.ci uses eps = min(1e-08, mean(t)/1e+06), so equal negative values are not constant
    assert not _boot_const(np.array([-0.3, -0.3, -0.3]))


@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    params = {'random_seed': 1, 'indy_var': '',
              'method': 'perc',
              'num_iterations': 100, 'event_equal': 'False',
              'agg_stat_input': f'{cwd}/data/agg_eclv_data.data',
              'agg_stat_output': f'{cwd}/data/agg_eclv_data_output.data',
              'fixed_vars_vals_input': {},
              'series_val_1': {'model': ['WRF'], 'fcst_lev': ['Z10', 'P850-700']},
              'alpha': 0.05, 'line_type': 'ctc',
              'num_threads': -1,
              'indy_vals': [],
              'agg_stat1': ['ECLV'],
              'circular_block_bootstrap': True,
              'equalize_by_indep': 'True',
              'cl_step': 0.05,
              'log_dir': f'{cwd}/logs/',
              'log_filename': 'log_agg_eclv.txt',
              'log_level': 'WARNING'
              }
    agg_stat = AggEclv(params)
    settings_dict = dict()
    settings_dict['agg_stat'] = agg_stat
    return settings_dict
