import datetime as dt
import pytest

from metcalcpy.validate_mv_python import get_testing_period, replace_name


def test_get_testing_period():
    param = {'start_date': '2020-01-01', 'end_date': '2020-01-16'}
    (start, end) = get_testing_period(param)
    assert start == dt.datetime.strptime('2020-01-01', '%Y-%m-%d').date()
    assert end == dt.datetime.strptime('2020-01-16', '%Y-%m-%d').date()

    param = {}
    (start, end) = get_testing_period(param)
    assert start == (dt.datetime.now() - dt.timedelta(1)).date()
    assert end == dt.datetime.now().date()

    param = {'start_date': '2020-01-01'}
    (start, end) = get_testing_period(param)
    assert start == dt.datetime.strptime('2020-01-01', '%Y-%m-%d').date()
    assert end == dt.datetime.now().date()

    param = {'end_date': '2020-01-01'}
    with pytest.raises(Exception):
        get_testing_period(param)


def test_replace_name():
    assert replace_name('plot_20200115_135714.xml', 'py') == 'plot_20200115_135714_py.xml'
