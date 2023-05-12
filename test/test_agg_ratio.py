import numpy as np
import pytest

from metcalcpy.agg_stat import AggStat, pd


def test_agg_ratio(settings):
    agg_stat = settings['agg_stat']
    agg_stat.calculate_stats_and_ci()
    result_frame = pd.read_csv(
        agg_stat.params['agg_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.size == 24
    assert result_frame.shape == (3, 8)
    assert np.allclose(result_frame['stat_value'][0], 3.15696)
    assert np.allclose(result_frame['stat_value'][1], 3.52230)
    assert np.allclose(result_frame['stat_value'][2], 0.89628)



@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    params = {'random_seed': 1, 'indy_var': 'fcst_lead',
              'method': 'perc',
              'num_iterations': 10, 'event_equal': 'True',
              'agg_stat_input': 'data/agg_ratio.data',
              'agg_stat_output': 'data/agg_ratio_data_output.data',
              'fixed_vars_vals_input': {
                  'obtype': {
                      'obtype_0' : ['CCPA']
                  },
                  'interp_mthd': {
                      'interp_mthd_3': ['NEAREST']
                  },
                  'vx_mask': {
                      'vx_mask_1': ['CONUS']
                  },
                  'fcst_init_beg': {
                      'fcst_init_beg_2': ['2022-04-30 00:00:00', '2022-05-01 00:00:00', '2022-05-02 00:00:00','2022-05-03 00:00:00', '2022-05-04 00:00:00',
                                          '2022-05-05 00:00:00',  '2022-05-06 00:00:00', '2022-05-07 00:00:00', '2022-05-08 00:00:00', '2022-05-09 00:00:00',
                                          '2022-05-10 00:00:00','2022-05-11 00:00:00', '2022-05-12 00:00:00' ]
                  }
              },
              'series_val_1': {'model': ['RRFSE_CONUS_ICperts_nostoch.rrfs_conuscompact_3km']},
              'series_val_2': {},
              'alpha': 0.05, 'line_type': 'ecnt',
              'num_threads': -1,
              'indy_vals': ['30000'],
              'circular_block_bootstrap': True,
              'equalize_by_indep': 'True',
              'cl_step': 0.05,
              'derived_series_1':[
                  ['RRFSE_CONUS_ICperts_nostoch.rrfs_conuscompact_3km APCP_03 ECNT_RMSE',
                   'RRFSE_CONUS_ICperts_nostoch.rrfs_conuscompact_3km APCP_03 ECNT_SPREAD',
                   'RATIO']
              ],
              'fcst_var_val_1':{
                  'APCP_03': ['ECNT_RMSE','ECNT_SPREAD']
              },
              'list_stat_1':['ECNT_RMSE', 'ECNT_SPREAD'],
              'list_stat_2':[]
              }
    agg_stat = AggStat(params)
    settings_dict = dict()
    settings_dict['agg_stat'] = agg_stat
    return settings_dict
