"""
Program Name: goodness_distance.py
This simple routine was patterned after the calc_speed routine in met_stats.py.
"""
import math
import numpy as np

__author__ = 'Jonathan Vigh'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


def calc_goodness_distance(sr, pod):
    """ Calculated the Euclidean distance between a model's location in the performance diagaram and the upper
        hand corner (perfect forecast)
        Args:
            sr: success_ratio (1 - False Alarm Ratio)
            pod: probabilty of detection

        Returns:
            "goodness" distance on the performance diagram or None
    """
    try:
        result = np.sqrt(sr * sr + pod * pod)
    except (TypeError, Warning):
        result = None
    return result
