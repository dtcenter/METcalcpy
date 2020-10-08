import pytest
import numpy as np
import metcalcpy.util.utils as utils

def test_lon_from_360_to_180():

    # Verify that longitude type is maintained, ie an int isn't converted to
    # a float when converting coord values.
    i_lon = [i_lon for i_lon in range (0, 359)]
    np_lon = np.linspace(0, 359, 360)

    i_west_east = utils.convert_lon_360_to_180(i_lon)
    np_west_east = utils.convert_lon_360_to_180(np_lon)


    assert isinstance(i_west_east[0], np.int64)
    assert isinstance(np_west_east[0], np.float)


    # Verify that some values were correctly converted
    # and that the ordering in the array is from negative to positive
    assert np_west_east[0] == -180.0
    assert np_west_east[359] == 179.0


if __name__ == "__main__":
    test_lon_from_360_to_180()




