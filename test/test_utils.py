"""Tests the operation of METcalcpy's utils code."""
import numpy as np
from metcalcpy.util.utils import represents_int, is_string_integer, get_derived_curve_name, calc_derived_curve_value, \
    unique, intersection, is_derived_series, parse_bool


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


def test_is_derived_series():
    series = ['analog_e', '3000', 'FBAR']
    assert not is_derived_series(series)
    series.append('DIFF')
    assert is_derived_series(series)


def test_parse_bool():
    assert parse_bool("True")
    assert not parse_bool("False")


if __name__ == "__main__":
    test_represents_int()
    test_is_string_integer()
    test_get_derived_curve_name()
    test_calc_derived_curve_value()
    test_unique()
    test_intersection()
    test_is_derived_series()
    test_parse_bool()
