import pytest
import numpy as np

from metcalcpy.scorecard import Scorecard, pd


def test_calculate_scorecard_data(settings):
    scorecard = settings['scorecard']
    scorecard.calculate_scorecard_data()
    result_frame = pd.read_csv(
        scorecard.params['sum_stat_output'],
        header=[0],
        sep='\t'
    )
    assert result_frame.size == 72
    assert result_frame.shape == (9, 8)
    assert np.allclose(
    result_frame[(result_frame['model'] == 'DIFF(P200 AFWAOCv3.5.1_d01 120000:240000 HGT BCMSE-P200 NoahMPv3.5.1_d01 120000:240000 HGT BCMSE)')
                 &(result_frame['vx_mask'] == 'EAST')
                 & (result_frame['fcst_lead'] == '120000:240000')]['stat_value'],-6.29828)

    assert np.allclose(
        result_frame[(result_frame[
                          'model'] == 'DIFF_SIG(P200 AFWAOCv3.5.1_d01 360000 HGT BCMSE-P200 NoahMPv3.5.1_d01 360000 HGT BCMSE)')
                     & (result_frame['vx_mask'] == 'NMT')
                     & (result_frame['fcst_lead'] == '120000:240000')]['stat_value'], -0.76703)

    assert np.allclose(
        result_frame[(result_frame[
                          'model'] == 'SINGLE(P200 AFWAOCv3.5.1_d01 480000 HGT BCMSE-P200 NoahMPv3.5.1_d01 480000 HGT BCMSE)')
                     & (result_frame['vx_mask'] == 'EAST')
                     & (result_frame['fcst_lead'] == '480000')]['stat_value'], 439.42705)




@pytest.fixture
def settings():
    """Initialise values for testing.

    Returns:
        dictionary with values of different type
    """
    params = {'append_to_file': False,
              'derived_series_1':
                  [['P200 AFWAOCv3.5.1_d01 120000:240000 HGT BCMSE',
                    'P200 NoahMPv3.5.1_d01 120000:240000 HGT BCMSE',
                    'DIFF'],
                   ['P200 AFWAOCv3.5.1_d01 360000 HGT BCMSE',
                    'P200 NoahMPv3.5.1_d01 360000 HGT BCMSE',
                    'DIFF_SIG'],
                   ['P200 AFWAOCv3.5.1_d01 480000 HGT BCMSE',
                    'P200 NoahMPv3.5.1_d01 480000 HGT BCMSE',
                    'SINGLE']
                   ],
              'equalize_by_indep': True,
              'event_equal': False,
              'fcst_var_val_1': {
                  'HGT':
                      ['BCMSE']},
              'fix_val_list_eq': [],
              'fixed_vars_vals_input': {},
              'indy_plot_val': [],
              'indy_vals': ['EAST',
                            'NMT',
                            'WEST',
                            'G2/TRO'],
              'indy_var': 'vx_mask',
              'line_type': 'sl1l2',
              'list_stat_1':
                  ['BCMSE'],
              'ndays': 10,
              'series_val_1': {
                  'fcst_lead': [
                      '120000:240000',
                      '360000',
                      '480000'],
                  'fcst_lev':
                      ['P200'],
                  'model':
                      ['AFWAOCv3.5.1_d01',
                       'NoahMPv3.5.1_d01']},
              'stat_flag': 'NCAR',
              'sum_stat_input': 'data/scorecard.data',
              'sum_stat_output': 'data/scorecard_output.data'

              }
    scorecard = Scorecard(params)
    settings_dict = dict()
    settings_dict['scorecard'] = scorecard
    return settings_dict
