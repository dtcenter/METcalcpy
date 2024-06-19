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

Examples for how to use this script can be found in the driver scripts of the use cases 
listed below.

* `Stratosphere Polar <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s/UserScript_fcstGFS_obsERA_StratospherePolar.html#sphx-glr-generated-model-applications-s2s-userscript-fcstgfs-obsera-stratospherepolar-py>`_
* `Blocking <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s_mid_lat/UserScript_fcstGFS_obsERA_Blocking.html#sphx-glr-generated-model-applications-s2s-mid-lat-userscript-fcstgfs-obsera-blocking-py>`_
* `Weather Regime <https://metplus.readthedocs.io/en/latest/generated/model_applications/s2s_mid_lat/UserScript_fcstGFS_obsERA_WeatherRegime.html#sphx-glr-generated-model-applications-s2s-mid-lat-userscript-fcstgfs-obsera-weatherregime-py>`_

Information about Input Data
============================

At this time, all input arrays have to be one dimensional only and should be the same size.  
The script does not make an attempt to check if input arrays are the same size.  If any of 
your input arrays are larger than the observation input array, the data will be chopped at 
the length of the observation input.  If an array is shorter than the observation input, the 
program will error.

Currently, the the following variables cannot be set and will be output as NA: FCST_THRESH, 
OBS_THRESH, COV_THRESH, ALPHA, OBS_QC, CLIMO_MEAN, CLIMO_STDEV, CLIMO_CDF.  Additionally the 
following variables also cannot be set and have default values: INTERP_MTHD = NEAREST, 
INTERP_PNTS =  1, and OBTYPE = ADPUPA.

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
    desc: 1D array string
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

  * metcalcpy

  * numpy

  * os
 
.. code-block:: ini

   write_mpr_file(data_fcst,data_obs,lats_in,lons_in,fcst_lead,fcst_valid,obs_lead,obs_valid,mod_name,desc,fcst_var,fcst_unit,fcst_lev,obs_var,obs_unit,obs_lev,maskname,obsslev,outdir,outfile_prefix)

The output fill be a .stat file located in outdir with data in `MET's Matched Pair Format <https://met.readthedocs.io/en/latest/Users_Guide/point-stat.html#id24>`_.  The file will be labeled with outfile_prefix and then have lead time, valid YYYYMMDD, and valid HHMMSS stamped onto the file name.
