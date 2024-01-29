***********
Aggregation
***********

Python Requirements
===================

The third-party Python packages and the corresponding version numbers are found
in the requirements.txt and nco_requirements.txt files:


Retrieve Code
=============

Refer to the Installation Guide for instructions.
The

Install Package
===============

Refer to the Installation Guide for instructions.


Aggregation
===========

The agg_stat module, **agg_stat.py** to is used to calculate aggregated statistics and confidence intervals.
This module can be run as a script at the command-line, or imported in another Python script.

A required YAML configuration file,  **config_agg_stat.yaml** files is used to define the location of
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

1.1 - Specify the input and output files

.. code-block:: yaml

  agg_stat_input: ./grid_stat_reformatted.agg.txt
  agg_stat_output: ./output.txt

1.2 - Specify the meteorological and the stat variables:

.. code-block:: yaml

  fcst_var_val_1:
    APCP_03:
    - FBIAS

1.3 - Specify the selected models/members:

.. code-block:: yaml

  series_val_1:
    model:
    - RRFS_GDAS_GF.SPP.SPPT_mem01
    - RRFS_GDAS_GF.SPP.SPPT_mem02
    - RRFS_GDAS_GF.SPP.SPPT_mem03

The full **config_agg_stat.yaml** file can be seen below:

.. code-block:: yaml

  agg_stat_input: ./grid_stat_reformatted.agg.txt
  agg_stat_output: ./output.txt
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

2. Run the python script:

.. code-block:: yaml

  bash
  python agg_stat.py config_stat_agg.yaml


The command above will generate a file called **output.txt** with the aggregated data that 
can be later plot using the METplotpy tools.


