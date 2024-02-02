import numpy as np
import pytest

from metcalcpy.util import sl1l2_statistics as sl1l2

def test_calculate_bcmse():
    # Test that negative BCMSE values are no longer returned.
    input_data_list = []

    # These data produce negative BCMSE values. Test that the modified code no longer returns negative values
    # for the BCMSE.
    input_data_list.append(np.array([[4.37978400e+01, 4.70115800e+01, 1.91825108e+03, 2.21008843e+03, 2.05900571e+03, 1.00000000e+00]]))
    input_data_list.append(np.array([[8.66233900e+01, 4.83037900e+01, 7.50361146e+03, 2.33325660e+03, 4.18423840e+03, 1.00000000e+00]]))
    input_data_list.append(np.array([[3.68089000e+01, 1.64253370e+02, 1.35489535e+03, 2.69791703e+04, 6.04598647e+03, 1.00000000e+00]]))
    columns_names = np.array(['fbar', 'obar', 'ffbar', 'oobar', 'fobar', 'total'], dtype='<U5')

    for input in input_data_list:
      result = sl1l2.calculate_bcmse(input, columns_names, aggregation=False)
      assert result >= 0.



