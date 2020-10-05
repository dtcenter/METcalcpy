# -*- coding: utf-8 -*-
"""
Piecewise linear function class.

"""

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Bill Campbell (NRL)'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


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

    def plot(self, x=np.nan, y=np.nan):
        """Plot the envelope function and knots."""
        fig, ax = plt.subplots(figsize=(15, 10))
        # Set the tick labels font
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(18)
            label.set_fontweight('normal')
        # Set the font dictionaries (for plot title and axis titles)
        title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black',
                      'weight': 'bold', 'verticalalignment': 'bottom'}
        axis_font = {'fontname': 'Arial', 'size': '18', 'weight': 'bold'}
        plt.xlabel('X Domain ({})'.format(self.xunits), **axis_font)
        plt.ylabel('Y Range', **axis_font)
        plt.title('Piecewise Linear Test ' + self.name, **title_font)
        plt.plot(self.x_domain, self.y_range, 'b-', linewidth=2)
        plt.plot(self.x_domain, self.y_range, 'bo', markersize=15)
        plt.plot(x, y, 'r*', markersize=20)
        plt.grid(True)
        plt.show()


def plot_envelope_only(xvals, yvals, ttype, xunits='ft'):
    """Plot the envelope function and knots."""
    fig, ax = plt.subplots(figsize=(15, 10))
    # Set the tick labels font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(18)
        label.set_fontweight('normal')
    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black',
                  'weight': 'bold', 'verticalalignment': 'bottom'}
    axis_font = {'fontname': 'Arial', 'size': '18', 'weight': 'normal'}
    plt.xlabel('Mean {} ({})'.format(ttype, xunits), **axis_font)
    plt.ylabel('Weighting on difficulty index', **axis_font)
    plt.title('Envelope for {}'.format(ttype), **title_font)
    plt.plot(xvals, yvals, 'r-', linewidth=4)
    #plt.plot(self.x_domain, self.y_range, 'bo', markersize=15)
    plt.xlim((xvals[0], xvals[-1]))
    plt.xticks(xvals)
    plt.ylim((np.min(yvals)-0.01, np.max(yvals)+0.11))
    idx = np.where(yvals == np.max(yvals))[0]
    xdata = np.tile(xvals[idx], (2, 1))
    yzeros = np.zeros_like(yvals[idx])
    ydata = np.array([yzeros, yvals[idx]])
    plt.plot(xdata, ydata, 'b--')
    plt.show()

    return fig, ax


def main():
    """Test envelope functions."""
    plt.close('all')
    xunits = 'feet'
    xmin = 0.0
    xmax = 25.0
    xinput = (xmax - xmin) * np.random.rand(5, 3, 2) - xmin
    print("Unsorted inputs (in {}) are:\n {}".format(xunits, xinput))

    # Envelope for versions 4 and 5
    A45_left = 1.0
    A45_right = 0.0
    A45_xlist = [12.0, 21.0]
    A45_ylist = [1.0, 0.0]
    A45 = PiecewiseLinear(A45_xlist, A45_ylist, xunits=xunits,
                          right=A45_right, left=A45_left, name="A4 & A5")
    A45_youtput = A45.values(xinput)
    print("Corresponding values are:\n {}".format(A45.values(xinput)))
    # Pass 1D views of x and y for plotting
    A45.plot(xinput.ravel(), A45_youtput.ravel())

    # Envelope for version 6
    A6_left = 1.0
    A6_right = 0.0
    A6_xlist = [6.0, 9.0, 12.0, 21.0]
    A6_ylist = [1.0, 1.5, 1.5, 0.0]
    A6 = PiecewiseLinear(A6_xlist, A6_ylist, xunits=xunits,
                         right=A6_right, left=A6_left, name="A6")
    A6_youtput = A6.values(xinput)
    print("Corresponding values are:\n {}".format(A6.values(xinput)))
    # Pass 1D views of x and y for plotting
    A6.plot(xinput.ravel(), A6_youtput.ravel())

    # Envelope for version 6.1
    A6_1_left = 0.0
    A6_1_right = 0.0
    A6_1_xlist = [3.0, 9.0, 12.0, 21.0]
    A6_1_ylist = [0.0, 1.5, 1.5, 0.0]
    A6_1 = PiecewiseLinear(A6_1_xlist, A6_1_ylist, xunits=xunits,
                           right=A6_1_right, left=A6_1_left, name="A6.1")
    A6_1_youtput = A6_1.values(xinput)
    print("Corresponding values are:\n {}".format(A6_1.values(xinput)))
    # Pass 1D views of x and y for plotting
    A6_1.plot(xinput.ravel(), A6_1_youtput.ravel())

    # Envelope for version 7
    A7_name = "A7"
    A7_left = 0.0
    A7_right = 0.0
    A7_xlist = [3.0, 9.0, 12.0, 21.0]
    A7_ylist = [0.0, 1.0, 1.0, 0.0]
    A7 = PiecewiseLinear(A7_xlist, A7_ylist, xunits=xunits,
                         right=A7_right, left=A7_left, name=A7_name)
    A7_youtput = A7.values(xinput)
    print("Corresponding values are:\n {}".format(A7.values(xinput)))
    # Pass 1D views of x and y for plotting
    A7.plot(xinput.ravel(), A7_youtput.ravel())

    # Produce envelope figure for documentation
    xvals = np.array([0.0] + A6_1_xlist + [24.0])
    yvals = np.array([0.0] + A6_1_ylist + [0.0])
    xunits = 'ft'
    ttype = 'significant wave height'
    fig1, ax1 = plot_envelope_only(xvals, yvals, ttype, xunits)
    fig1.savefig('./Figure_1.png')

    # Produce envelope figure for documentation
    xvals = np.array([0.0, 5.0, 28.0, 34.0, 50.0, 55.0])
    yvals = np.array([0.0, 0.0,  1.5,  1.5,  0.0,  0.0])
    xunits = 'kts'
    ttype = 'wind speed'
    fig2, ax2 = plot_envelope_only(xvals, yvals, ttype, xunits)
    fig2.savefig('./Figure_2.png')

    return fig1, ax1, fig2, ax2


if __name__ == "__main__":
    FIG1, AX1, FIG2, AX2 = main()
