# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
# -*- coding: utf-8 -*-
"""
Piecewise linear function class.

"""

import numpy as np

__author__ = 'Bill Campbell (NRL)'
__version__ = '0.1.0'


class IncompatibleLengths(Exception):
    """Custom exception for PiecewiseLinear input checking."""


class UnsortedArray(Exception):
    """Custom exception for PiecewiseLinear input checking."""


class PiecewiseLinear():
    """
    Defines a piecewise linear function with a given domain and range.

    Xdomain is a numpy array of knot locations. Yrange is a numpy array.
    """

    def __init__(self, x_domain, y_range, xunits='feet',
                 left=np.nan, right=np.nan, name=""):
        len_x = len(x_domain)
        if len_x < 2:
            raise IncompatibleLengths('Length of xdomain must be at least 2.')
        if np.any(np.diff(x_domain)) < 0:
            print("X_domain (in {}) is {}".format(xunits, x_domain))
            raise UnsortedArray('Xdomain must be sorted in ascending order.')
        len_y = len(y_range)
        if len_x != len_y:
            raise IncompatibleLengths('X_domain and Y_range must have same ' +
                                      'length.\n Use left and right to set ' +
                                      'value for points outside the x_domain\n')
        self.x_domain = np.array(x_domain)
        self.y_range = np.array(y_range)
        self.xunits = xunits
        self.left = left
        self.right = right
        self.name = name

    def get_ymax(self):
        """Find maximum of envelope function"""
        return np.max(self.y_range)

    def values(self, xinput):
        """
        Evaluate piecewise linear function for the set of points in xinput.

        xinput is a set of points inside xdomain.
        """
        if not isinstance(xinput, np.ndarray):
            xinput = np.array(xinput)
        xin_shape = np.shape(xinput)
        # Treat xinput as one-dimensional for use by np.interp
        xview = xinput.ravel()
        yflat = np.interp(xview, self.x_domain, self.y_range,
                          left=self.left, right=self.right)
        # Restore output to same shape as input
        youtput = np.reshape(yflat, xin_shape)

        return youtput

if __name__ == "__main__":
    pass
