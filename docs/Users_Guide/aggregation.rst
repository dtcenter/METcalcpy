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

The agg_stat module support the following linetypes that are output from the MET
**point-stat** and **grid-stat** tools:

* CTC
* SL1L2
* SAL1L2
* VAL1L2
* VCNT
* PSTD
* MCTS
* PCT

In addition, the following linetypes from the MET **grid-stat** tool are supported:

* GRAD
* NBRCNT
* NBRCTC

Finally, the following linetypes from the MET **ensemble-stat** tool are supported:

* SSVAR (the SSVAR_SPREAD and SSVAR_RMSE statistics are exempt)
* ECNT

In order to aggregate the filtered data (**grid_stat_reformatted.agg.txt**) produced above,
it is necessary to edit the settings in the **config_agg_stat.yaml** file:

Modify the YAML configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The config_agg_stat.yaml configuration file is located in the $METCALCPY_BASE/metcalcpy/pre_processing/aggregation/config
directory. The $METCALCPY_BASE is the directory where the METcalcpy source code is
saved (e.g. /Users/my_acct/METcalcpy). Change directory to $METCALCPY_BASE/metcalcpy/pre_processing/aggregation/config
and modify the config_agg_stat.yaml file.

1.  Specify the input and output files

.. code-block:: yaml

  agg_stat_input: /path-to/rrfs_cts_reformatted.data
  agg_stat_output: /path-to/rrfs_cts_aggregated.txt

Replace the *path-to* in the above two settings to the location where the input data
was stored (either in a working directory or the $METCALCPY_BASE/test directory). **NOTE**:
Use the **full path** to the input and output directories (no environment variables).

2.  Specify the meteorological and the stat variables:

.. code-block:: yaml

  fcst_var_val_1:
    APCP_03:
    - FBIAS

3.  Specify the selected models/members:

.. code-block:: yaml

  series_val_1:
    model:
    - RRFS_GDAS_GF.SPP.SPPT_mem01
    - RRFS_GDAS_GF.SPP.SPPT_mem02
    - RRFS_GDAS_GF.SPP.SPPT_mem03

The full **config_agg_stat.yaml** file is shown below:

.. code-block:: yaml

  agg_stat_input: /Users/my_account/sample_data/rrfs_cts_reformatted.data
  agg_stat_output: /Users/my_account/my_output/rrfs_cts_aggregated.data
  alpha: 0.05
  append_to_file: null
  circular_block_bootstrap: 'True'
  derived_series_1: []
  derived_series_2: []
  event_equal: 'FALSE'
  fcst_var_val_1:
    APCP_03:
    - FBIAS
  fcst_var_val_2: {}
  indy_vals:
  - '30000'
  - '60000'
  - '90000'
  - '120000'
  - '150000'
  - '180000'
  - '210000'
  - '240000'
  - '270000'
  - '300000'
  - '330000'
  - '360000'
  indy_var: fcst_lead
  line_type: ctc
  list_stat_1:
  - FBIAS
  list_stat_2: []
  method: perc
  num_iterations: 1
  num_threads: -1
  random_seed: null
  series_val_1:
    model:
    - RRFS_GDAS_GF.SPP.SPPT_mem01
    - RRFS_GDAS_GF.SPP.SPPT_mem02
    - RRFS_GDAS_GF.SPP.SPPT_mem03
  series_val_2: {}


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


This will generate the file **rrfs_cts_aggregated.data** which contains the
aggregated statistics data that can be used to generate plots using METplotpy.



Additionally, the agg_stat.py module can be invoked by another script or module
by importing the package:

.. code-block:: ini

  from metcalcpy.agg_stat import AggStat

  AGG_STAT = AggStat(PARAMS)
  AGG_STAT.calculate_stats_and_ci()

where PARAMS is a dictionary containing the parameters indicating the
location of input and output data. The structure is similar to the
original Rscript template from which this Python implementation was derived.

Use the same PYTHONPATH defined above to ensure that the agg_stat module is found by
the Python import process.