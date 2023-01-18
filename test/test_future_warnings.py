import pytest
import itertools
import re
import pandas as pd
import numpy as np


from metcalcpy.util.utils import equalize_axis_data, aggregate_field_values, event_equalize
from metcalcpy.agg_stat_event_equalize import AggStatEventEqualize
from metcalcpy.util import pstd_statistics as pstd
from metcalcpy.agg_stat_eqz import AggStatEventEqz
from metcalcpy import GROUP_SEPARATOR, DATE_TIME_REGEX

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
    settings_dict['series_val_2'] = {}
    settings_dict['indy_var'] = 'fcst_lead'
    settings_dict['line_type'] = None


    return settings_dict

def test_equalize_axis_data(settings):
    '''
        Test that the FutureWarning is no longer generated when invoking the util.utils.equalize_axis_data() function
    '''
    print("Testing equalize_axis_data with 'dummy' event equalize data for FutureWarning...")
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


@pytest.fixture
def settings_agg_stat():
    """Initialise values for testing agg_stat_ee.

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
              'series_val_2': {'model': ['ENS001v3.6.1_d01', 'ENS001v3.6.1_d02']},
              'alpha': 0.05, 'line_type': 'sl1l2',
              'num_threads': -1,
              'indy_vals': ['0', '30000', '60000', '90000',
                            '120000', '150000', '180000', '210000', '240000'],
              'list_stat_1': ['FBAR'],
              'list_stat_2': [],
              'circular_block_bootstrap': False}
    agg_stat_ee = AggStatEventEqualize(params)
    settings_dict = dict()
    settings_dict['agg_stat_input'] = agg_stat_ee

    return settings_dict


def test_calculate_values(settings_agg_stat):
    '''

       Test that FutureWarning is no longer being raised in the agg_stat_event_equalize module at ~line 710 in
       the calculate_values() function.
    '''
    print("Testing calculate_values in agg_stat_event_equalize with agg_stat_and_boot data for FutureWarning...")


    # This test fails if the some_dataframe.append(extra_df) wasn't replaced with pd.concat(some_dataframe, extra_df)
    # Around line 708 of util.utils module's equalize_axis_data function.
    try:
        asee =  settings_agg_stat['agg_stat_input']
        asee.calculate_values()
    except FutureWarning:
        assert False


def test_run_ee_on_axis(settings_agg_stat):
    '''

       Test that FutureWarning is no longer being raised in the agg_stat_event_equalize module at ~line 148 in
       the run_ee_on_axis() function.
    '''
    print("Testing run_ee_on_axis in agg_stat_event_equalize with agg_stat_and_boot data for FutureWarning...")


    # This test fails if the some_dataframe.append(extra_df) wasn't replaced with pd.concat(some_dataframe, extra_df)
    # Around line 708 of util.utils module's equalize_axis_data function.
    try:
        asee =  settings_agg_stat['agg_stat_input']
        fix_vals_permuted = [('P100',)]
        asee.run_ee_on_axis(fix_vals_permuted)
    except FutureWarning:
        assert False


def test_calculate_pstd_roc_auc():
    '''

       Test the calculate_pstd_roc_auc to ensure no FutureWarnings are present
    '''
    columns = np.array(['fcst_var','thresh_i','on_i', 'oy_i'])
    # 'thresh_i': [], 'oy_i': [], 'on_i': []
    # input_file = "data/reliability.data"
    input_file = "data/roc_sample.data"
    cur_df = pd.read_csv(input_file, sep='\t')
    input_data = np.array(cur_df)
    try:
       pstd.calculate_pstd_roc_auc(input_data, columns)
    except FutureWarning:
        assert False

@pytest.fixture
def settings_ee_dummy():
    settings_dict = dict()
    settings_dict['fix_val_keys'] = []
    settings_dict['fcst_var_val_1'] = dict({'TC': ["BASER", "CTC"]})
    settings_dict['fcst_var_val_2'] = {}
    settings_dict['fix_vals_permuted'] = {}
    settings_dict['series_val_1'] = dict({'model': ["GFSDCF;"]})
    settings_dict['series_val_2'] = {}
    settings_dict['indy_var'] = 'fcst_lead'
    settings_dict['line_type'] = None

    return settings_dict


def test_aggregate_field_values(settings_ee_dummy):
    '''
       Test aggregation on field value that is NOT fcst_lead (to exercise one branch of the "if")
       to ensure no FutureWarnings are present (~line 513)
    '''
    input_file = "data/event_equalize_dummy.data"
    cur_df = pd.read_csv(input_file, sep='\t')
    series_var_val = {}
    series_var_val['series_val_1'] = settings_ee_dummy['series_val_1']
    series_var_val['series_val_2'] = settings_ee_dummy['series_val_2']
    line_type = settings_ee_dummy['line_type']
    try:
        aggregate_field_values(settings_ee_dummy['series_val_1'], cur_df, line_type)
        assert True
    except FutureWarning:
        assert False

@pytest.fixture
def settings_ee_dummy2():
    settings_dict = dict()
    settings_dict['fix_val_keys'] = []
    settings_dict['fcst_var_val_1'] = dict({'TC': ["BASER", "CTC"]})
    settings_dict['fcst_var_val_2'] = {}
    settings_dict['fix_vals_permuted'] = {}
    settings_dict['series_val_1'] = dict({'fcst_lead': ['240000;']})
    settings_dict['series_val_2'] = {}
    settings_dict['indy_var'] = 'fcst_lead'
    settings_dict['line_type'] = ''

    return settings_dict


def test_aggregate_field_values_agg_by_fcst_lead(settings_ee_dummy2):
    '''
        Test with aggregation based on fcst_lead to exercise the "else"
        to ensure no FutureWarnings are present (~line 562)
    '''
    input_file = "data/event_equalize_group_input.data"
    cur_df = pd.read_csv(input_file, sep='\t')
    series_var_val = {}
    series_var_val['series_val_1'] = settings_ee_dummy2['series_val_1']
    series_var_val['series_val_2'] = settings_ee_dummy2['series_val_2']
    line_type = settings_ee_dummy2['line_type']
    try:
        aggregate_field_values(settings_ee_dummy2['series_val_1'], cur_df, line_type)
        assert True
    except FutureWarning:
        assert False

