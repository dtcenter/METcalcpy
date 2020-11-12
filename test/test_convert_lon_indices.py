import pytest
import numpy as np
import metcalcpy.util.utils as utils

def test_convert_lon_indices_working():

    # Verify that some values were correctly converted
    # and that the ordering in the array is from negative to positive
    np_lon = np.linspace(0, 359, 360)
    minlon_in = -180
    range_in = 360
    west_east, west_east_indices = utils.convert_lons_indices(np_lon, minlon_in, range_in)
    assert west_east[0] == -180.0
    assert west_east[359] == 179.0


if __name__ == "__main__":
    test_convert_lon_indices_working()
