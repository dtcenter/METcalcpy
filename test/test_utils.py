"""Tests the operation of METcalcpy's utils code."""
import numpy as np
import pandas as pd
import pytest

from metcalcpy.util.utils import represents_int, is_string_integer, get_derived_curve_name, calc_derived_curve_value, \
    unique, intersection, is_derived_point, parse_bool, round_half_up, sum_column_data_by_name, \
    nrow_column_data_by_name_value, create_permutations_mv, column_data_by_name, calculate_mtd_revision_stats, \
    autocor_coef, is_string_strictly_float


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


def test_represents_int():
    assert represents_int(1)
    assert not represents_int("1")
    assert not represents_int(1.5)
    assert not represents_int(None)


def test_is_string_integer():
    assert is_string_integer("1")
    assert not is_string_integer("1.5")
    assert not is_string_integer("not_int")
    assert not is_string_integer(None)


def test_is_string_float():
    assert not is_string_strictly_float("1")
    assert is_string_strictly_float("1.5")
    assert not is_string_strictly_float("not_float")
    assert not is_string_strictly_float("not.float")
    assert not is_string_strictly_float(None)


def test_get_derived_curve_name():
    list_of_names = ['analog_e GHI FBAR', 'dicast15 GHI FBAR']
    assert 'DIFF(analog_e GHI FBAR-dicast15 GHI FBAR)' == get_derived_curve_name(list_of_names)
    list_of_names.append('RATIO')
    assert 'RATIO(analog_e GHI FBAR/dicast15 GHI FBAR)' == get_derived_curve_name(list_of_names)


def test_calc_derived_curve_value():
    val1 = np.array([674.08, 100])
    val2 = np.array([665.97, 100])
    result = calc_derived_curve_value(val1, val2, 'DIFF')
    assert np.allclose(np.array([8.11, 0.]), result)
    result = calc_derived_curve_value(val1, val2, 'RATIO')
    assert np.allclose(np.array([1.01217773, 1.]), result)
    val2[1] = 0
    result = calc_derived_curve_value(val1, val2, 'RATIO')
    assert not result


def test_unique():
    in_list = ['analog_e', 'GHI', 'FBAR', '3000', 'FBAR', '3000']
    out_list = unique(in_list)
    out_list.sort()
    assert ['3000', 'FBAR', 'GHI', 'analog_e'] == out_list
    assert not unique(None)


def test_intersection():
    l_1 = ['a', 'b', 'c', 'd']
    l_2 = ['g', 'b', 'o', 'd']
    assert ['b', 'd'] == intersection(l_1, l_2)
    assert not intersection(None, l_2)


def test_is_derived_point():
    point = ('stochmp1', '20000', 'PSTD_BRIER')
    assert not is_derived_point(point)
    point = (
        'DIFF(stochmp1 TMP_ENS_FREQ_ge283 PSTD_BRIER-stochmp2 TMP_ENS_FREQ_ge283 PSTD_BRIER)', '20000', 'PSTD_BRIER')
    assert is_derived_point(point)


def test_parse_bool():
    assert parse_bool("True")
    assert not parse_bool("False")


def test_round_half_up():
    assert 1.2 == round_half_up(1.23, 1)
    assert 1.3 == round_half_up(1.28, 1)
    assert 1.3 == round_half_up(1.25, 1)
    assert -1.2 == round_half_up(-1.25, 1)
    assert 0.12346 == round_half_up(0.1234567875, 5)
    assert 456792.12346 == round_half_up(456792.1234567875, 5)


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
    assert not sum_column_data_by_name(data_values, settings['columns'], column_name, rm_none=False)
    assert 1425310.4448 == sum_column_data_by_name(data_values, settings['columns'], column_name, rm_none=True)


def test_column_data_by_name(settings):
    data_values = np.array([
        ['dicast15', '2019-07-03 12:00:00', '2019-07-04 09:00:00', 210000, 'SWS01', 'GHI', 'MAE', 0, 1, 1073.4, 1085.7,
         1165390.38, 1152187.56, 1178744.49, 12.3],
        ['dicast15', '2019-07-03 12:00:00', '2019-07-05 13:15:00', 491500, 'SWS01', 'GHI', 'MAE', 0, 1, 518.43, 501.36,
         259920.0648, 268769.6649, 251361.8496, 17.07]
    ])
    column_name = 'fobar'
    result = column_data_by_name(data_values, settings['columns'], column_name)
    assert len(result) == 2
    assert float(data_values[0, 11]) == result[0]
    assert float(data_values[1, 11]) == result[1]

    column_name = 'not_in_array'
    assert not column_data_by_name(data_values, settings['columns'], column_name)

    data_values = np.append(data_values,
                            [['dicast15', '2019-07-03 12:00:00', '2019-07-04 09:00:00', 210000, 'SWS01', 'GHI', 'MAE',
                              0, 1, 1073.4, 1085.7, None, 1152187.56, 1178744.49, 12.3]], axis=0)
    column_name = 'fobar'
    result = column_data_by_name(data_values, settings['columns'], column_name)
    assert len(result) == 3
    assert np.isnan(result[2])

    result = column_data_by_name(data_values, settings['columns'], column_name, rm_none=True)
    assert len(result) == 2


def test_nrow_column_data_by_name_value(settings):
    data_values = np.array([
        ['dicast15', '2019-07-03 12:00:00', '2019-07-04 09:00:00', 210000, 'SWS01', 'GHI', 'MAE', 0, 1, 1073.4, 1085.7,
         1165390.38, 1152187.56, 1178744.49, 12.3],
        ['dicast15', '2019-07-03 12:00:00', '2019-07-05 13:15:00', 491500, 'SWS01', 'GHI', 'MAE', 0, 1, 518.43, 501.36,
         259920.0648, 268769.6649, 251361.8496, 17.07],
        ['dicast15', '2019-07-03 12:00:00', '2019-07-05 13:15:00', 491500, 'SWS01', 'GHI', 'MAE', 0, 1, 518.43, 501.36,
         259920.0648, 268769.6649, 251361.8496, 17.07],
        ['dicast15', '2019-07-03 12:00:00', '2019-07-05 13:15:00', 491500, 'SWS01', 'GHI', 'MAE', 1, 1, 518.43, 501.36,
         259920.0648, 268769.6649, 251361.8496, 17.07]
    ])
    filters = {'fcst_lead': 491500, 'stat_value': 0}
    assert 2 == nrow_column_data_by_name_value(data_values, settings['columns'], filters)


def test_create_permutations_mv_dict():
    fields_values = {'vx_mask': ['FULL'], 'model': ['HREF', 'HREFV3'], 'fcst_var': ['HGT'],
                     'stat_name': ['ECNT_RMSE', 'ECNT_SPREAD']}
    expected_result = [['FULL', 'HREF', 'HGT', 'ECNT_RMSE'], ['FULL', 'HREFV3', 'HGT', 'ECNT_RMSE'],
                       ['FULL', 'HREF', 'HGT', 'ECNT_SPREAD'], ['FULL', 'HREFV3', 'HGT', 'ECNT_SPREAD']]
    result = create_permutations_mv(fields_values, 0)
    assert expected_result == result


def test_create_permutations_mv_dict_empty():
    fields_values = {'vx_mask': ['FULL'], 'model': ['HREF', 'HREFV3'], 'fcst_var': [],
                     'stat_name': ['ECNT_RMSE', 'ECNT_SPREAD']}
    expected_result = [['FULL', 'HREF', 'ECNT_RMSE'], ['FULL', 'HREFV3', 'ECNT_RMSE'],
                       ['FULL', 'HREF', 'ECNT_SPREAD'], ['FULL', 'HREFV3', 'ECNT_SPREAD']]
    result = create_permutations_mv(fields_values, 0)
    assert expected_result == result


def test_create_permutations_mv_list():
    fields_values = [['FULL'], ['HREF', 'HREFV3'], ['HGT'], ['ECNT_RMSE', 'ECNT_SPREAD']]
    expected_result = [['FULL', 'HREF', 'HGT', 'ECNT_RMSE'], ['FULL', 'HREFV3', 'HGT', 'ECNT_RMSE'],
                       ['FULL', 'HREF', 'HGT', 'ECNT_SPREAD'], ['FULL', 'HREFV3', 'HGT', 'ECNT_SPREAD']]
    result = create_permutations_mv(fields_values, 0)
    assert expected_result == result


def test_calculate_mtd_revision_stats():
    df = pd.read_csv('data/mtd_revision.data', index_col=0)
    stats = calculate_mtd_revision_stats(df)
    assert stats.get("ww_run") == 0
    assert stats.get("auto_cor_p") == 0
    assert stats.get("auto_cor_r") == 0.01
    stats = calculate_mtd_revision_stats(df, 1)
    assert stats.get("auto_cor_r") == 0.15

def test_autocor_coef():
    z = [28, 28, 26, 19, 16, 24, 26, 24, 24, 29, 29, 27, 31, 26, 38, 23, 13, 14, 28, 19, 19, 17, 22, 2, 4, 5, 7, 8, 14, 14, 23]
    result = autocor_coef(z)
    assert round_half_up(result, 7) == 0.6635721

if __name__ == "__main__":
    test_represents_int()
    test_is_string_integer()
    test_is_string_float()
    test_get_derived_curve_name()
    test_calc_derived_curve_value()
    test_unique()
    test_intersection()
    test_is_derived_point()
    test_parse_bool()
    test_round_half_up()
    test_sum_column_data_by_name()
    test_calculate_mtd_revision_stats()
    test_autocor_coef()
