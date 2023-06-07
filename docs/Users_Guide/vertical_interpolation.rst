**********************
Vertical Interpolation
**********************

Description
===========

This module is used to perform pressure to height conversion in TC-RMW
data (netCDF or grb2) by vertically interpolating fields
between grids with pressure vertical coordinates.  The pressure to height conversion is
implemented with linear interpolation.


Example
=======

**Sample Data**

Sample TC-RMW data (in pressure level) is located:

https://dtcenter.ucar.edu/dfiles/code/METplus/METplotpy/tcrmw/tc_rmw_example.nc.gz

Save this data file to a directory of your choosing and *cd* to your directory:

``cd $METCALCPY_DATA_DIR``

*$METCALCPY_DATA_DIR* is the directory where you stored the example data.

Uncompress the data using *gunzip*:

``gunzip tc_rmw_example.nc.gz``

You should now have a file: *tc_rmw_example.nc*


**Configuration Files**

A configuration file (YAML, with a .yaml extension) is used to define which variables in the
input data file (netCDF or grib2) are to be converted from pressure levels to height:

**height_from_pressure_tcrmw.yaml**:

.. literalinclude:: ../../examples/height_from_pressure_tcrmw.yaml

This configuration file is located in the $METCALCPY_SOURCE_DIR/examples/ directory, where
*$METCALCPY_SOURCE_DIR* is the location of where you saved the METcalcpy source code.

In this example, the UGRD, VGRD, and TMP variables are selected for conversion
(refer to the `fields` setting in the above configuration file).  You may list any other
variables as long as they are present in the input data:

e.g.

fields:
    - 'UGRD'
    - 'VGRD'
    - 'TMP'
    - 'RH'
    - 'PRMSL'



Run from the Command Line
=========================

A sample Bourne-shell script can be used to convert the pressure level data to height level data:

$METCALCPY_SOURCE_DIR/METcalcpy/examples/height_from_pressure_tcrmw.sh


.. literalinclude:: ../../examples/height_from_pressure_tcrmw.sh


You will need to set the environment variable for the location of the data directory
(where the input data resides), the name of the input data, the name of the YAML configuration file,
and the name of the output file (output directory and filename).  NOTE: If you do not want to see
the debug messages, simply delete the line with the `--debug`.  Open the *height_from_pressure_tcrmw.sh*
file using an editor of your choice.

Replace the `/path/to/input-data` to the actual full path to
the directory where you saved the sample data:

e.g.

``export DATA_DIR=/users/mydir/data/tcrmw``

Replace the `/path/to/output` to the directory where you want to save your output:

e.g.

``--output /users/mydir/tcrmw/output``

Where in this example, the /users/mydir/tcrmw/output is the directory where the output should be directed.  Replace this with the full path to
the desired location for output files.

Uncomment the `--debug` if additional debug information is desired.  This will result in the generation of intermediate netCDF files in the directory specified by the $DATA_DIR 
environment variable in the `height_from_pressure_tcrmw.sh` shell script.  Create a 'Debug' subdirectory in the $DATA_DIR directory.  After running the
`height_from_pressure_tcrmw.sh` shell script, numerous netCDF files will be created: a height_from_pressure_debug.nc and numerous files beginning with `vertical_interp_debug`
 	   
Save and close the file.

To perform the conversion, do the following:

* Make sure you have these required Python packages:

  * Python 3.7

  * metpy 1.1.0

  * netcdf4 1.5.7

  * numpy 1.21.2

  * pint 0.18
 
  * pyyaml 5.4.1

  * xarray 0.20.1

  * yaml 0.2.5

* If running in a conda environment, verify that you are running in a conda environment that
  has the above Python packages installed.

* cd to the $METCALCPY_SOURCE_DIR/METcalcpy/examples directory

* run the script, enter the following command:

``sh height_from_pressure_tcrmw.sh``

This will produce a netCDF file with the filename specified in the *height_from_pressure_tcrmw.sh* script,
in this case it is *tc_rmw_example_vertical_interp.nc* and will be located in the output directory specified
via the $OUTPUT_DIR environment variable.  This file contains the converted levels for the
fields specified in the *height_from_pressure_tcrmw.yaml* configuration file.





