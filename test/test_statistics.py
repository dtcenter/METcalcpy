"""Tests the operation of METcalcpy's statistics code."""
import numpy as np
import pytest
from metcalcpy.util.statistics import get_column_index_by_name


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


if __name__ == "__main__":
    test_get_column_index_by_name()
