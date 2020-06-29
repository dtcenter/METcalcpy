"""Tests the operation of METcalcpy's event_equalize code."""
import itertools
import time

import pandas as pd
import pytest

from metcalcpy.event_equalize import event_equalize
from metcalcpy.util.utils import represents_int


def test_event_equalize():
    """Tests event equalization."""

    indy_var = "fcst_lead"
    indy_vals = ["30000", "60000", "90000"]
    series_val = dict({'model': ["GFSDCF", "GFSRAW"]})
    fixed_vars_vals_input = dict({
        'fcst_thresh': dict({'fcst_thresh_4': ["<=20"]}),

    })

    list_stat = ['BASER']
    list_static_val = dict({'fcst_var': 'APCP_06'})
    fcst_var_val = dict({'TCDC': ["BASER"]})
    input_data_file = 'data/event_equalize_input.data'
    output_data_file = 'data/event_equalize_output_py.data'
    cl_step = 0.05
    bool_multi = False

    start_all = time.time()
    # read the input data file into a data frame
    input_data = pd.read_csv(input_data_file, header=[0], sep='\t')

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
                    split_val = val.split(',')
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

    # save to file
    output_data.to_csv(index=False, sep='\t', path_or_buf=output_data_file)
    end_all = time.time()
    print("total :" + str(end_all - start_all))


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


if __name__ == "__main__":
    test_event_equalize()
