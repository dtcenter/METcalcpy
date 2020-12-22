# tropical-diagnostics
Python scripts for tropical diagnostics of NWP forecasts.

The diagnostics are meant to be applied to gridded forecast data and example scripts are provided to show how
to apply the diagnostics at different lead times.

Required model output is primarily precipitation. This is enough to compute Hovmoeller diagrams and
compare to observations and to project onto the convectively coupled equatorial wave (CCEW) EOFs to
analyze CCEW activity and skill in model forecasts.

To compute cross-spectra between precipitation and dynamical variables, single level dynamical fields are also
required. In the example coherence spectra between divergence (850hPa or 200hPa) and precipitation are considered.

## tropical_diagnostics
Contains the functions and modules necessary to compute the various diagnostics. The main diagnostics
included are:

### Hovmoeller diagrams
Functions to compute hovmoeller latitudinal averages and pattern correlation are included in
**hovmoeller_calc.py**. Plotting routines are included in **hovmoeller_plotly.py**. The driver script is
**example_hovmoeller.py**, this reads in data from the default data directory ../data and computes latitude
averages and calls the plotting routine.

### Space-time spectra
Functions for computing 2D Fourier transforms and 2D power and cross-spectra are included in **spacetime.py**.
To plot the spectra **spacetime_plot.py** uses pyngl, which is based on NCL and provides similar control
over plotting resources. The driver script is **example_cross_spectra.py**, this reads in data from the default
data directory ../data and computes cross-spectral estimates. The output is saved as netcdf and the output directory
needs to be specified. The driver script for plotting is **example_cross_spectra_plot.py**, this reads the output
files from **example_cross_spectra.py** and calls the plotting routine.

### CCEW activity and skill
Functions to project precipitation (either from model output or observations) onto CCEW EOF patterns and
compute wave activity and a CCEW skill score are included in **ccew_activity.py**. Also included are routines
to plot the activity and the skill compared to observations. The EOFs are provided on a 1 degree lat-lon grid, the
path to the location of the EOF files needs to be specified in the driver script **example_kelvin_activity.py**. This
script calls the routines to compute the activity index and the plotting routines.