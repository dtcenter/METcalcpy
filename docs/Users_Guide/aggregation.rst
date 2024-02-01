***********
Aggregation
***********

Aggregation is an option that can be applied to MET stat output (in
the appropriate format) to calculate aggregation statistics and confidence intervals.
Input data must first be reformatted using the METdataio METreformat module to
label all the columns with the corresponding statistic name specified in the
`MET User's Guide <https://met.readthedocs.io/en/develop/Users_Guide/index.html>`_
for `point-stat <https://met.readthedocs.io/en/develop/Users_Guide/point-stat.html>`_,
`grid-stat <https://met.readthedocs.io/en/develop/Users_Guide/grid-stat.html>`_, or
`ensemble-stat <https://met.readthedocs.io/en/develop/Users_Guide/ensemble-stat.html>`_ .stat output data.

Python Requirements
===================

The third-party Python packages and the corresponding version numbers are found
in the requirements.txt and nco_requirements.txt files:

**For Non-NCO systems**:

* `requirements.txt <https://github.com/dtcenter/METcalcpy/blob/develop/requirements.txt>`_

**For NCO systems**:

* `nco_requirements.txt <https://github.com/dtcenter/METcalcpy/blob/develop/nco_requirements.txt>`_


Retrieve Code
=============

Refer to the `Installation Guide <https://metcalcpy.readthedocs.io/en/develop/Users_Guide/installation.html>`_
for instructions.


Retrieve Sample Data
====================

The sample data used for this example is located in the $METCALCPY_BASE/test directory,
where **$METCALCPY_BASE** is the full path to the location of the METcalcpy source code
(e.g. /User/my_dir/METcalcpy).
The example data file used for this example is **rrfs_ecnt_for_agg.data**.
This data was reformatted from the MET .stat output using the METdataio METreformat module.
The reformatting step labels the columns with the corresponding statistics, based on the MET tool (point-stat,
grid-stat, or ensemble-stat).  The ECNT linetype of
the MET grid-stat output has been reformatted to include the statistics names for all
`ECNT <https://met.readthedocs.io/en/develop/Users_Guide/ensemble-stat.html#id2>`_ specific columns.


Input data **must** be in this format prior to using the aggregation
module, agg_stat.py.

The example data can be copied to a working directory, or left in this directory.  The location
of the data will be specified in the YAML configuration file.

Please refer to the METdataio User's Guide for instructions for reformatting MET .stat files :
https://metdataio.readthedocs.io/en/develop/Users_Guide/reformat_stat_data.html


Aggregation
===========

The agg_stat module, **agg_stat.py** to is used to calculate aggregated statistics and confidence intervals.
This module can be run as a script at the command-line, or imported in another Python script.

A required YAML configuration file,  **config_agg_stat.yaml** file is used to define the location of
input data and the name and location of the output file.

The agg_stat module support the ECNT linetype that are output from the MET
**ensemble-stat** tool

The input to the agg_stat module must have the appropriate format.  The ECNT linetype must first be
`reformatted via the METdataio METreformat module <https://metdataio.readthedocs.io/en/develop/Users_Guide/reformat_stat_data.html>`_
by following the instructions under the **Reformatting for computing aggregation statistics with METcalcpy agg_stat**
header.

Modify the YAML configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The config_agg_stat.yaml is required to perform aggregation statistics calculations. This
configuration file is located in the $METCALCPY_BASE/metcalcpy/pre_processing/aggregation/config
directory. The $METCALCPY_BASE is the directory where the METcalcpy source code is
saved (e.g. /Users/my_acct/METcalcpy). Change directory to $METCALCPY_BASE/metcalcpy/pre_processing/aggregation/config
and modify the config_agg_stat.yaml file.

1.  Specify the input and output files

.. code-block:: yaml

  agg_stat_input: /path-to/test/data/rrfs_ecnt_for_agg.data
  agg_stat_output: /path-to/ecnt_aggregated.data

Replace the *path-to* in the above two settings to the location where the input data
was stored (either in a working directory or the $METCALCPY_BASE/test directory). **NOTE**:
Use the **full path** to the input and output directories (no environment variables).

2.  Specify the meteorological and the stat variables:

.. code-block:: yaml

  fcst_var_val_1:
    TMP:
      - ECNT_RMSE
      - ECNT_SPREAD_PLUS_OERR

3.  Specify the selected models/members:

.. code-block:: yaml

  series_val_1:
    model:
     - RRFS_GEFS_GF.SPP.SPPT

4.  Specify the selected statistics to be aggregated, in this case, the RMSE and SPREAD_PLUS_OERR
    statistics from the ECNT ensemble-stat tool output are to be calculated.  The aggregated statistics
    are named ECNT_RMSE and ECNT_SPREAD_PLUS_OERR (append original statistic name with the linetype):

    list_stat_1:
     - ECNT_RMSE
     - ECNT_SPREAD_PLUS_OERR

The full **config_agg_stat.yaml** file is shown below:


.. literalinclude:: ../../metcalcpy/pre_processing/aggregation/config/config_agg_stat.yaml



**NOTE**: Use full directory paths when specifying the location of the input file and output
file.


Set the Environment and PYTHONPATH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bash shell:

.. code-block:: ini

 export METCALCPY_BASE=/path-to-METcalcpy

csh shell:

.. code-block:: ini

 setenv METCALCPY_BASE /path-to-METcalcpy


where *path-to-METcalcpy* is the full path to where the METcalcpy source code is located
(e.g. /User/my_dir/METcalcpy)

bash shell:

.. code-block:: ini

 export PYTHONPATH=$METCALCPY_BASE/:$METCALCPY_BASE/metcalcpy

csh shell

.. code-block:: ini

 setenv PYTHONPATH $METCALCPY_BASE/:$METCALCPY_BASE/metcalcpy


Where $METCALCPY_BASE is the full path to where the METcalcpy code resides (e.g. /User/
my_dir/METcalcpy).

Run the python script:
^^^^^^^^^^^^^^^^^^^^^^

The following are instructions for performing aggregation from the command-line:

.. code-block:: yaml


  python $METCALCPY_BASE/metcalcpy/agg_stat.py $METCALCPY_BASE/metcalcpy/pre_processing/aggregation/config/config_stat_agg.yaml


This will generate the file **ecnt_aggregated.data** (from the agg_stat_output setting) which now contains the
aggregated statistics data. This data is in a format that can be read by the METplotpy line plot
to generate a spread-skill plot by plotting the ECNT_RMSE and ECNT_SPREAD_PLUS_OERR.



Additionally, the agg_stat.py module can be invoked by another script or module
by importing the package:

.. code-block:: ini

  from metcalcpy.agg_stat import AggStat

  AGG_STAT = AggStat(PARAMS)
  AGG_STAT.calculate_stats_and_ci()

where PARAMS is a dictionary containing the parameters indicating the
location of input and output data. The structure is similar to the
original Rscript template from which this Python implementation was derived.

**NOTE**: Remember to use the same PYTHONPATH defined above to ensure that the agg_stat module is found by
the Python import process.