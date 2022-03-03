import pandas as pd
import os

from metcalcpy import GROUP_SEPARATOR
from metcalcpy.agg_stat import AggStat


def test_groups():
    # prepare parameters
    params = {'random_seed': 1, 'indy_var': 'fcst_lead',
              'method': 'perc',
              'num_iterations': 1, 'event_equal': 'True',
              'derived_series_1': [
                  ['Group_y1_1 TMP ME', 'Group_y1_2 TMP ME', 'DIFF']],
              'derived_series_2': [],
              'agg_stat_input': 'data/agg_stat_with_groups_data.data',
              'fcst_var_val_1': {'TMP': ['ME']},
              'fcst_var_val_2': {},
              'agg_stat_output': 'data/agg_stat_with_groups_output.data',
              'fixed_vars_vals_input': {'fcst_lev': {'fcst_lev_0': ['Z02']}},
              'series_val_1': {'model': ['GTS+RAIN3mm' + GROUP_SEPARATOR + 'GTS+RAIN4mm',
                                         'GTS+RAIN50p' + GROUP_SEPARATOR + 'GTS+RAIN5mm']},
              'series_val_2': {},
              'alpha': 0.05, 'line_type': 'sl1l2',
              'num_threads': -1,
              'indy_vals': ['0', '120000', '240000'],
              'list_stat_1': ['ME'],
              'list_stat_2': []}
    # start aggregation logic
    AGG_STAT = AggStat(params)
    AGG_STAT.calculate_stats_and_ci()

    # read the output
    input_data = pd.read_csv(
        params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert len(input_data) == 9
    assert input_data.loc[(input_data['model'] == 'GTS+RAIN3mm' + GROUP_SEPARATOR + 'GTS+RAIN4mm')
                          & (input_data['fcst_lead'] == 120000)]['stat_value'].item() == -1.4384707
    assert input_data.loc[(input_data['model'] == 'GTS+RAIN50p' + GROUP_SEPARATOR + 'GTS+RAIN5mm')
                          & (input_data['fcst_lead'] == 240000)]['stat_value'].item() == -1.5292608
    assert input_data.loc[(input_data['model'] == 'DIFF(Group_y1_1 TMP ME-Group_y1_2 TMP ME)')
                          & (input_data['fcst_lead'] == 0)]['stat_value'].item() == -0.02839
    # remove the output
    os.remove(params['agg_stat_output'])
