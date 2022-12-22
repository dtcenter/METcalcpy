import pytest
import pandas as pd
import numpy as np
import warnings
from metcalcpy.util.utils import equalize_axis_data

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

    settings_dict['fix_val_keys'] = []
    settings_dict['fcst_var_val_1'] = dict({'TCDC': ["BASER", "CTC"],'TC': ["BASER", "CTC"] })
    settings_dict['fcst_var_val_2'] = {}
    settings_dict['fix_vals_permuted'] = {}
    settings_dict['series_val_1'] = dict({'model': ["GFSDCF"]})
    settings_dict['indy_var'] = 'fcst_lead'
    settings_dict['line_type'] = None


    return settings_dict

def test_equalize_axis_data(settings):
    '''
        Test that the FutureWarning is no longer generated when invoking the util.utils.equalize_axis_data() function
    '''
    print("Testing equalize_axis_data with event equalize data for FutureWarning...")
    input_file = "data/event_equalize_dummy.data"
    cur_df = pd.read_csv(input_file, sep='\t')
    fix_vals_keys = []
    fix_vals_permuted_list = []

    # This test fails if the some_dataframe.append(extra_df) wasn't replaced with pd.concat(some_dataframe, extra_df)
    # Around line 708 of util.utils module's equalize_axis_data function.
    try:
        ee_df = equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, settings, cur_df, axis='1')
    except FutureWarning:
        assert False


