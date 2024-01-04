**********************
Write MPR
**********************

Description
===========

This program writes data to an output file in MET’s Matched Pair (MPR) format.  It 
takes several inputs, which are described in the list below.  The script will compute 
the observation input and total number of observations.  It will also check to see if 
the output directory is present and will create that directory if it does not exist.

Example
=======

Examples for how to use this script can be found in the driver scripts of multiple use 
cases listed below.
`Stratosphere Bias <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s/UserScript_fcstGEFS_Difficulty_Index.html#sphx-glr-generated-model-applications-medium-range-userscript-fcstgefs-difficulty-index-py>`_.
`Stratosphere Polar Cap <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s/UserScript_fcstGEFS_Difficulty_Index.html#sphx-glr-generated-model-applications-medium-range-userscript-fcstgefs-difficulty-index-py>`_.
`Blocking <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s_mid_lat/UserScript_fcstGFS_obsERA_Blocking.html#sphx-glr-generated-model-applications-s2s-mid-lat-userscript-fcstgfs-obsera-blocking-py>`_.
`Weather Regime <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s_mid_lat/UserScript_fcstGFS_obsERA_WeatherRegime.html#sphx-glr-generated-model-applications-s2s-mid-lat-userscript-fcstgfs-obsera-weatherregime-py>`_.

Information about Input Data
============================

At this time, all input arrays have to be one dimensional only and should be the same size.  The script does not make an attempt to check if input arrays are the same size.  If any of your input arrays are larger than the observation input array, the data will be chopped at the length of the observation input.  If an array is shorter than the observation input, the program will error.

Currently, the the following variables cannot be set and will be output as NA: FCST_THRESH, OBS_THRESH, COV_THRESH, ALPHA, OBS_QC, CLIMO_MEAN, CLIMO_STDEV, CLIMO_CDF.  Additionally the following variables also cannot be set and have default values: INTERP_MTHD = NEAREST, INTERP_PNTS =  1, and OBTYPE = ADPUPA.

    data_fcst: 1D array float
            forecast data to write to MPR file
    data_obs: 1D array float
            observation data to write to MPR file
    lats_in: 1D array float
            data latitudes
    lons_in: 1D array float
            data longitudes
    fcst_lead: 1D array string of format HHMMSS
            forecast lead time
    fcst_valid: 1D array string of format YYYYmmdd_HHMMSS
            forecast valid time
    obs_lead: 1D array string of format HHMMSS
            observation lead time
    obs_valid: 1D array string of format YYYYmmdd_HHMMSS
            observation valid time
    mod_name: string
            output model name (the MODEL column in MET)
    desc: string
            output description (the DESC column in MET)
    fcst_var: 1D array string
            forecast variable name
    fcst_unit: 1D array string
            forecast variable units
    fcst_lev: 1D array string
            forecast variable level
    obs_var: 1D array string
            observation variable name
    obs_unit: 1D array string
            observation variable units
    obs_lev: 1D array string
            observation variable level
    maskname: string
            name of the verification masking region
    obsslev: 1D array string
            Pressure level of the observation in hPA or accumulation
            interval in hours
    outdir: string
            Full path including where the output data should go
    outfile_prefix: string
            Prefix to use for the output filename.  The time stamp will
            be added in MET's format based off the first forecast time



Run from a python script
=========================


* Make sure you have these required Python packages:

  * Python 3.7

  * metpy 1.1.0

  * netcdf4 1.5.7

  * numpy 1.21.2

  * pint 0.18
 
  * pyyaml 5.4.1

  * xarray 0.20.1

  * yaml 0.2.5

.. code-block:: ini

  sh height_from_pressure_tcrmw.sh

This will produce a netCDF file with the filename specified in the *height_from_pressure_tcrmw.sh* script,
in this case it is *tc_rmw_example_vertical_interp.nc* and will be located in the output directory specified
via the $OUTPUT_DIR environment variable.  This file contains the converted levels for the
fields specified in the *height_from_pressure_tcrmw.yaml* configuration file.





