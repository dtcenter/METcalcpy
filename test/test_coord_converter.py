import pytest
import numpy as np
from metcalcpy.util.utils import convert_coords


def test_coord_converter():

    # Verify that coord type is maintained, ie an int isn't converted to
    # a float when converting coord values.
    i_lon = [i_lon for i_lon in range (0, 359)]
    np_lon = np.linspace(0, 359, 360)

    i_west_east = convert_coords(i_lon)
    np_west_east = convert_coords(np_lon)


    assert isinstance(i_west_east[0], np.int64)
    assert isinstance(np_west_east[0], np.float)

    # Verify that some values were correctly converted
    assert np_west_east[359] == -1.0
    assert np_west_east[179] == 179.0

if __name__ == "__main__":
    test_coord_converter()




