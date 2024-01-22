# aggregation (Aggregation Workflow)


## Use Case

### Reformatting and Filtering

For this step the files aggregation_preprocessor.py and config_aggregation_preprocessor.yaml files are required.

1 - Modify the variables prefix, suffix, dates, and members in the config_aggregation_preprocessor.yaml file to point to the data you wish to process:

For instance, if the variables are set as the following:

```
prefix: "/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/"
suffix: "/metprd/grid_stat_cmn"
dates:
- '2022050100'
- '2022050200'
- '2022050300'
members:
- 'mem01'
- 'mem02'
- 'mem03'
```

the code will look for the data (.stat files) in the following folders:

```
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050100/mem01/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050100/mem02/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050100/mem03/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050200/mem01/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050200/mem02/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050200/mem03/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050300/mem01/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050300/mem02/metprd/grid_stat_cmn
/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/2022050300/mem03/metprd/grid_stat_cmn
```

2 - (Optional) It is possible to group the entire dataset using the following:

```
group_members: True
group_name: "RRFS_GDAS_GF.SPP_agg"
```

3 - It is also necessary to select the meteorological variable (APCP_03, APCP_01, TMP...), the threshold and the stat variable (FBAR, FBIAS, GSS...):

```
fcst_var:
- APCP_03
fcst_thresh:
- ">0.0"
list_stat: 
- FBAR
```

4 - Additional settings might need to be adjusted as well:

```
prefix: "/scratch2/BMC/fv3lam/HIWT/expt_dirs/RRFS_GDAS_GF.SPP.SPPT_20220501-06/"
suffix: "/metprd/grid_stat_cmn"
dates:
- '2022050100'
- '2022050200'
- '2022050300'
members:
- 'mem01'
- 'mem02'
- 'mem03'
group_members: False
group_name: "RRFS_GDAS_GF.SPP_agg"
output_xml_file: "point_stat.xml"
output_yaml_file: "point_stat.yaml"
output_reformatted_file: "grid_stat_reformatted.txt"
output_aggregate_file: "grid_stat_reformatted.agg.txt"
metdataio_dir: "/path/to/METdataio" 
fcst_var:
- APCP_03
fcst_thresh:
- ">0.0"
list_stat: 
- FBIAS
log_file: log.agg_wflow
```

5 - You can set a WORK_DIR folder and copy the required files to it before executing the python script. To execute the python script use the following command:

```bash
python aggregation_preprocessor.py -y config_aggregation_preprocessor.yaml
```

Considering the settings above, the command will create two output files: 

- grid_stat_reformatted.txt : File containing the reformatted data
- grid_stat_reformatted.agg.txt : Filtered data that can be used by agg_stat.py

### Aggregation

For this step the files agg_stat.py and config_agg_stat.yaml files are required.

1 - In order to aggregate the filtered data (grid_stat_reformatted.agg.txt) produced above, it is necessary to edit the settings in the config_agg_stat.yaml file:

1.1 - Specify the input and output files
```
agg_stat_input: ./grid_stat_reformatted.agg.txt
agg_stat_output: ./output.txt
```

1.2 - Specify the meterological and the stat variables:
```
fcst_var_val_1:
  APCP_03:
  - FBIAS
```

1.3 - Specify the selected models/members:
```
series_val_1:
  model:
  - RRFS_GDAS_GF.SPP.SPPT_mem01
  - RRFS_GDAS_GF.SPP.SPPT_mem02
  - RRFS_GDAS_GF.SPP.SPPT_mem03
```

The full config_agg_stat.yaml file can be seen below:

```
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
```

2 - Run the python script:

```bash
python agg_stat.py config_stat_agg.yaml
```

The command above will generate a file called output.txt with the aggregated data that can be later plot using the METplotpy tools.


### Plot with METplotpy

For this step the files line.py, config_plot_cmn.yaml and custom_line.yaml files are required.

config_plot_cmn.yaml : Config file containing common settings across the different plot types.
custom_line.yaml : Config file specific for the line plot.

1 - The yaml_preprocessor.py file is responsible for combining config_plot_cmn.yaml with the custom config file for the specific plot, in this case custom_line.yaml.

```bash
python yaml_preprocessor.py config_plot_cmn.yaml custom_line.yaml -o config_line.yaml
```

The command above will create the config_line.yaml file which is the result of the combination of the both config files config_plot_cmn.yaml custom_line.yaml, where custom_line.yaml variables have priority over the config_plot_cmn.yaml variables.

2 - Creating the line plot

```bash
python line.yaml config_line.yaml 
```

### Aggregation workflow

Additionally, a python wrapper named aggregation_WE2E.py is available to run all the steps mentioned above at once.
Specify the settings using the environment.yaml file and the config files mentioned above since they are coppied to a WORK_DIR folder.

```bash
python aggregation_WE2E.py
```


