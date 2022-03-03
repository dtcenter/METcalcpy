"""Tests the operation of METcalcpy's statistics code."""
import numpy as np
import pytest
from metcalcpy.util.met_stats import get_column_index_by_name
from metcalcpy.util.correlation import corr, acf
from metcalcpy.util.utils import round_half_up
from metcalcpy.util.wald_wolfowitz_runs_test import runs_test
from metcalcpy.util.eclv_statistics import calculate_eclv


def test_get_column_index_by_name(settings):
    column_name = 'fobar'
    assert 11 == get_column_index_by_name(settings['columns'], column_name)

    column_name = 'not_in_array'
    assert not get_column_index_by_name(settings['columns'], column_name)


@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    settings_dict = dict()
    columns = np.array(['model', 'fcst_init_beg', 'fcst_valid_beg', 'fcst_lead', 'vx_mask', 'fcst_var',
                        'stat_name', 'stat_value', 'total', 'fbar', 'obar', 'fobar', 'ffbar', 'oobar',
                        'mae'])
    settings_dict['columns'] = columns

    return settings_dict


def test_corr():
    x = [103.4, 59.92, 68.17, 94.54, 69.48, 72.17, 74.37, 84.44, 96.74, 94.26, 48.52, 95.68]
    y = [90.11, 77.71, 77.71, 97.51, 58.21, 101.3, 79.84, 96.06, 89.3, 97.22, 61.62, 85.8]

    corr_val = corr(x=x, y=y)['r'].tolist()[0]
    assert round_half_up(corr_val, 2) == 0.67


def test_acf():
    x = [2.4, 2.4, 2.4, 2.2, 2.1, 1.5, 2.3, 2.3, 2.5, 2.0, 1.9, 1.7, 2.2, 1.8, 3.2, 3.2, 2.7, 2.2, 2.2, 1.9, 1.9, 1.8,
         2.7, 3.0, 2.3, 2.0, 2.0, 2.9, 2.9, 2.7, 2.7, 2.3, 2.6, 2.4, 1.8, 1.7, 1.5, 1.4, 2.1, 3.3, 3.5, 3.5, 3.1, 2.6,
         2.1, 3.4, 3.0, 2.9]
    acf_val = acf(x, 'correlation', lag_max=10)
    assert -0.154 == round_half_up(acf_val[10], 3)
    acf_val = acf(x, 'correlation')
    assert 0.151 == round_half_up(acf_val[16], 3)


def test_eclv():
    x = np.array([[666, 112, 25, 33, 496], [350, 73, 9, 22, 246], [316, 39, 16, 11, 250]])
    columns_names = np.array(['total', 'fy_oy', 'fy_on', 'fn_oy', 'fn_on'])
    cl_pts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    thresh = 0
    line_type = 'ctc'
    add_base_rate = 1
    eclv = calculate_eclv(x, columns_names, thresh, line_type, cl_pts, add_base_rate)
    assert 0.7244291 == eclv['vmax']
    assert 0.0479846 == eclv['F']
    assert 0.7724138 == eclv['H']
    assert 0.2177177 == eclv['s']
    assert 20 == len(eclv['V'])
    assert -0.2514395393474087 == eclv['V'][0]



def test_runs_test():
    x = [2.4, 2.4, 2.4, 2.2, 2.1, 1.5, 2.3, 2.3, 2.5, 2.0, 1.9, 1.7, 2.2, 1.8, 3.2, 3.2, 2.7, 2.2, 2.2, 1.9, 1.9, 1.8,
         2.7, 3.0, 2.3, 2.0, 2.0, 2.9, 2.9, 2.7, 2.7, 2.3, 2.6, 2.4, 1.8, 1.7, 1.5, 1.4, 2.1, 3.3, 3.5, 3.5, 3.1, 2.6,
         2.1, 3.4, 3.0, 2.9]

    ww_run = runs_test(x, 'left.sided', 'median')
    assert 0.00117 == round_half_up(ww_run['p_value'], 5)


if __name__ == "__main__":
    test_get_column_index_by_name()
    test_corr()
    test_acf()
    test_runs_test()
    test_eclv()
