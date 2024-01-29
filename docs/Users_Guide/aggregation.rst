***********
Aggregation
***********

Aggregation is an option that can be applied to MET stat output (in
the appropriate format) to calculate aggregation statistics and confidence intervals.
Input data must first be reformatted using the METdataio METreformat module to
reorder the statistics and confidence limits into separate columns.

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

Refer to the `Installation Guide <https://metcalcpy.readthedocs.io/en/develop/Users_Guide/installation.html>`_ for instructions.


Retrieve Sample Data
====================

The sample data used for this example is located in the $METCALCPY_BASE/test directory,
where **$METCALCPY_BASE** is the full path to the location of the METcalcpy source code
(e.g. /User/my_dir/METcalcpy).
The example data file used for this example is **rrfs_cts_reformatted.data**.
This data was reformatted from the MET .stat output using the METdataio METreformat module.
The reformatting step collects the statistics and any confidence limits for a specified linetype.  The CTS linetype of
the MET grid-stat output has been reformatted into separate columns: stat_name, stat_value, stat_ncl,
stat_ncu, stat_bcl, and stat_bcu.  Input data **must** be in this format prior to using the aggregation
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

  agg_stat_input: ./rrfs_cts_reformatted.data
  agg_stat_output: ./rrfs_cts_aggregated.txt
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


In the configuration file above, the input data and output file will be located in the directory from
where the agg_stat.py script is run.


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


  python agg_stat.py config_stat_agg.yaml


This will generate the file **rrfs_cts_aggregated.txt** which contains the
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