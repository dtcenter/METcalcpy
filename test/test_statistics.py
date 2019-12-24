"""Tests the operation of METcalcpy's statistics code."""
import numpy as np
import pytest
from metcalcpy.util.statistics import get_column_index_by_name
from metcalcpy.util.utils import sum_column_data_by_name


def test_get_column_index_by_name(settings):
    column_name = 'fobar'
    assert 11 == get_column_index_by_name(settings['columns'], column_name)

    column_name = 'not_in_array'
    assert not get_column_index_by_name(settings['columns'], column_name)


def test_sum_column_data_by_name(settings):
    data_values = np.array([
        ['dicast15', '2019-07-03 12:00:00', '2019-07-04 09:00:00', 210000, 'SWS01', 'GHI', 'MAE', 0, 1, 1073.4, 1085.7,
         1165390.38, 1152187.56, 1178744.49, 12.3],
        ['dicast15', '2019-07-03 12:00:00', '2019-07-05 13:15:00', 491500, 'SWS01', 'GHI', 'MAE', 0, 1, 518.43, 501.36,
         259920.0648, 268769.6649, 251361.8496, 17.07]
    ])
    column_name = 'fobar'
    assert 1425310.4448 == sum_column_data_by_name(data_values, settings['columns'], column_name)

    column_name = 'not_in_array'
    assert not sum_column_data_by_name(data_values, settings['columns'], column_name)

    data_values = np.append(data_values,
                            [['dicast15', '2019-07-03 12:00:00', '2019-07-04 09:00:00', 210000, 'SWS01', 'GHI', 'MAE',
                              0, 1, 1073.4, 1085.7, None, 1152187.56, 1178744.49, 12.3]], axis=0)
    column_name = 'fobar'
    assert not sum_column_data_by_name(data_values, settings['columns'], column_name)


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


if __name__ == "__main__":
    test_get_column_index_by_name()
    test_sum_column_data_by_name()
