"""Tests the operation of METcalcpy's event_equalize code."""
import itertools
import time
import re

import pandas as pd
import pytest

from metcalcpy.event_equalize import event_equalize
from metcalcpy.util.utils import represents_int
from metcalcpy.util.utils import equalize_axis_data
from metcalcpy import GROUP_SEPARATOR, DATE_TIME_REGEX

def test_event_equalize():
    """Tests event equalization."""

    indy_var = "fcst_lead"
    series_val = dict({'model': ["GFSDCF", "GFSRAW"]})
    fixed_vars_vals_input = dict({
        'fcst_thresh': dict({'fcst_thresh_4': ["<=20"]}),

    })

    fcst_var_val = dict({'TCDC': ["BASER"]})
    input_data_file = 'data/event_equalize_input.data'

    # read the input data file into a data frame
    input_data = pd.read_csv(input_data_file, header=[0], sep='\t')

    output_data = perform_event_equalize(fcst_var_val, fixed_vars_vals_input,
                                         indy_var, input_data, series_val)

    assert len(input_data) == 244
    assert len(output_data) == 164

    # test groups

    indy_var = "fcst_lead"
    series_val = dict({'model': ["CONTROL:GTS"],
                       'fcst_valid_beg': ['2010-06-01 00:00:00:2010-06-01 12:00:00:2010-06-02 00:00:00']})
    fixed_vars_vals_input = dict()

    fcst_var_val = dict({'TMP': ["ME"]})
    input_data_file = 'data/event_equalize_group_input.data'

    # read the input data file into a data frame
    input_data = pd.read_csv(input_data_file, header=[0], sep='\t')
    output_data = perform_event_equalize(fcst_var_val, fixed_vars_vals_input,
                                         indy_var, input_data, series_val)

    assert len(input_data) == 216
    assert len(output_data) == 216


def perform_event_equalize(fcst_var_val, fixed_vars_vals_input, indy_var, input_data, series_val):
    cl_step = 0.05
    bool_multi = False
    start_all = time.time()

    # sort the dataset by init time, lead time and independent variable
    input_data.sort_values(by=['fcst_init_beg', 'fcst_lead', indy_var])
    fix_vars = []
    fix_vals = []
    output_data = pd.DataFrame()
    # list all fixed variables
    if fixed_vars_vals_input:
        for key, value in fixed_vars_vals_input.items():
            fix_vars.append(key)
            fix_vals.append(list(value.values()))
    # permute fix vals
    fix_vals_permuted = list(itertools.chain.from_iterable(fix_vals))
    # perform EE for each forecast variable
    for fcst_var, fcst_var_stats in fcst_var_val.items():
        for fcst_var_stat in fcst_var_stats:
            for series_var, series_var_vals in series_val.items():
                # ungroup series value
                series_var_vals_no_group = []
                for val in series_var_vals:
                    split_val = re.findall(DATE_TIME_REGEX, val)
                    if len(split_val) == 0:
                        split_val = val.split(GROUP_SEPARATOR)
                    series_var_vals_no_group.extend(split_val)

                series_data = input_data[(input_data['fcst_var'] == fcst_var)
                                         & (input_data["stat_name"] == fcst_var_stat)
                                         & (input_data[series_var].isin(series_var_vals_no_group))]
                start = time.time()
                series_data = \
                    event_equalize(series_data, indy_var, series_val, fix_vars,
                                   fix_vals_permuted, True, bool_multi)
                end = time.time()
                print("one EE:" + str(end - start))

                # append EE data to result
                if output_data.empty:
                    output_data = series_data
                else:
                    output_data.append(series_data)
    end_all = time.time()
    print("total :" + str(end_all - start_all))
    return output_data


@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    settings_dict = dict()
    settings_dict['int'] = 8
    settings_dict['str'] = 'str'
    settings_dict['date'] = '2019-09-05 03:00:00'
    settings_dict['double'] = 34.9
    settings_dict['int_n'] = -700
    return settings_dict

def test_represents_int_not_int(settings):
    """Tests that this fails to cast a type that cannot be cast into an int
        Args:
            settings: dictionary with values of different type
    """
    assert not represents_int(settings['str'])
    assert represents_int(settings['int'])
    assert represents_int(settings['int_n'])
    assert not represents_int(settings['double'])
    assert not represents_int(settings['date'])

@pytest.fixture
def settings_no_fix_vals():
    """

     Returns:
          settings_dict: a dictionary corresponding to the params arg for equalize_axis_data()
    """
    settings_dict = dict()
    settings_dict['fix_val_keys'] = []
    settings_dict['fcst_var_val_1'] = {}
    settings_dict['fcst_var_val_2'] = {}
    settings_dict['fix_vals_permuted'] = {}
    settings_dict['series_val_1'] = {}
    settings_dict['indy_var'] = {}
    settings_dict['line_type'] = None

    return settings_dict

def test_equalize_axis_data_no_fix_val(settings_no_fix_vals):
    '''
       conditions that lead to an empty data frame with no 'equalize' column following event equalization should
       return an empty data frame (instead of an error due to trying to clean up the 'equalize' column before
       returning the event equalization data frame).
    '''

    print("Testing equalize_axis_data with ROC CTC threshold data...")
    input_file_list = ["data/ROC_CTC_thresh.data", "data/ROC_CTC.data"]
    for input_file in input_file_list:
        cur_df = pd.read_csv(input_file, sep='\t')
        fix_vals_keys = []
        fix_vals_permuted_list = []

        ee_df = equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, settings_no_fix_vals, cur_df, axis='1')

        assert ee_df.shape[0] == 0


@pytest.fixture
def settings_no_fcst_var_vals():
    """

     Returns:
          settings_dict: a dictionary corresponding to the params arg for equalize_axis_data()
    """
    settings_dict = dict()
    settings_dict['fix_val_keys'] = []
    # No fcst_var_val in 'config'
    # settings_dict['fcst_var_val_1'] = {}
    # settings_dict['fcst_var_val_2'] = {}
    settings_dict['fix_vals_permuted'] = {}
    settings_dict['series_val_1'] = {}
    settings_dict['indy_var'] = ''
    settings_dict['line_type'] = None

    return settings_dict

def test_equalize_axis_data_no_fcst_var(settings_no_fcst_var_vals):
    '''
       Event equalization returns a data frame with no removed rows.
    '''

    print("Testing equalize_axis_data with ROC CTC threshold data...")
    input_file_list = ["data/ROC_CTC_thresh.data", "data/ROC_CTC.data"]
    for input_file in input_file_list:
        cur_df = pd.read_csv(input_file, sep='\t')
        fix_vals_keys = []
        fix_vals_permuted_list = []

        ee_df = equalize_axis_data(fix_vals_keys, fix_vals_permuted_list, settings_no_fcst_var_vals, cur_df, axis='1')
        assert ee_df.shape[0] == cur_df.shape[0]


if __name__ == "__main__":
    test_event_equalize()
