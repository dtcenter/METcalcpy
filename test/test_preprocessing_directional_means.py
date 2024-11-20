import pytest
import xarray as xr
import numpy as np
from metcalcpy.pre_processing import directional_means as dm

TEST_DATA = xr.Dataset(
    {
        "variable": xr.DataArray(
            [
                [1.1, 2.2, 3.3],
                [2.2, 3.3, 4.4],
                [4.4, 5.5, 6.6],
            ],
            coords={
    "latitude": [-10.25, -11.25, -12.25],
    "longitude": [112.25, 113.25, 114.25],
},
            dims=["latitude", "longitude"],
        ),
    },
    attrs={},
)


def test_zonal_mean():
    actual = dm.zonal_mean(TEST_DATA)
    np.testing.assert_almost_equal(actual.variable.values, np.array([2.2, 3.3, 5.5]))

    actual = dm.zonal_mean(TEST_DATA, "latitude")
    np.testing.assert_almost_equal(actual.variable.values, np.array([2.57, 3.67, 4.77]), 2)


def test_meridional_mean():
    actual = dm.meridional_mean(TEST_DATA, -12.25, -10.25)
    np.testing.assert_almost_equal(actual.variable.values, np.array([2.562829, 3.662829, 4.762829]))

    actual = dm.meridional_mean(TEST_DATA, 112.25, 114.25, "longitude")
    np.testing.assert_almost_equal(actual.variable.values, np.array([2.2297922, 3.3297922, 5.5297922]))


def test_meridional_mean_rasied():
    with pytest.raises(ValueError):
        dm.meridional_mean(TEST_DATA, -10.25, -10.25)

    with pytest.raises(ValueError):
        dm.meridional_mean(TEST_DATA, -10.25, -12.25)
