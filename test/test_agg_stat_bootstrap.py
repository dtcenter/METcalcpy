import os
import pathlib
import pandas as pd
import pytest

from metcalcpy.agg_stat_bootstrap import AggStatBootstrap
from metcalcpy.util.read_env_vars_in_config import parse_config


cwd = os.path.dirname(__file__)

def get_parms(config_file):
   """
   :param config_file:
   :return: dictionary representation of the yaml config file settings
   """
   os.environ['TEST_DIR'] = cwd
   return parse_config(config_file)

def cleanup(filename):
   """
     Clean up temporary output generated during the test

     :param filename: file to remove
     :return: None
   """

   pathlib.Path(filename).unlink()



def agg_stat_bootstrap(parms):
    """
      Calculate the aggregation statistics for the data specified in the configuration file.
        :param parms: The dictionary representation of configuration settings in the YAML config file.

        :return:
        the agg_stat_bootstrap.py script generates an output file of the statistic(s) specified in the YAML config file.


    """

    agg_stat_bootstrap_obj = AggStatBootstrap(parms)
    agg_stat_bootstrap_obj.calculate_values()

def test_calc_stats():
    """
      Verify that the _calc_stats function does not have any errors emanating from the refactoring of _calc_stats for
      logging and error handling updates.

      :return:
        None
    """

    config_file = f'{cwd}/data/mode/aggregate_with_python.yaml'
    parms = get_parms(config_file)

    try:
       agg_stat_bootstrap(parms)
    except SyntaxError:
        pytest.fail("Error still persists with _calc_stats code")

    # clean up
    output_file = parms['agg_stat_output']
    cleanup(output_file)

def test_compare_with_rscript():
    """
       Compare aggregation statistics results against those computed via R.
       Use METviewer to generate the aggregation statistics by de-selecting the 'use python'

       :return: None
   """

    # Use METcalcpy directly to calculate aggregation bootstrap statistics using
    # the same input data used by METviewer (SQL query results)

    config_file = f'{cwd}/data/mode/aggregate_with_python.yaml'
    parms = get_parms(config_file)
    assert os.path.isfile(config_file)
    infile = parms['agg_stat_input']
    df: pd.DataFrame = pd.read_csv(infile, sep=r'\s+')
    # Verify the input data exists
    assert not df.empty

    # Get the bootstrap stats calculated via the original R implementation (from METviewer)
    aggregated_via_r = f'{cwd}/data/mode/rscript_data/aggregated_by_rscript.data'
    r_df:pd.DataFrame = pd.read_csv(aggregated_via_r, sep=r'\s+')
    assert not r_df.empty

    # Invoke METcalcpy (i.e. Python implementation) to calculate the aggregation bootstrap statistics
    try:
       agg_stat_bootstrap(parms)
       # Check for the existence of the output with the requested aggregation statistics computed
       outfile = parms['agg_stat_output']
       print(f"outfile: {outfile}")
       assert os.path.isfile(outfile)
       # Compare stat_value results from R and Python for various fcst_lead
       # hours
       fcst_hrs = [0, 10000, 30000]
       for hr in fcst_hrs:
           # statistics calculated via R implementation
           r_fcst_row: pd.DataFrame = r_df[r_df['fcst_lead'] == hr]

           # statistics calculate via Python implementation
           df_python = pd.read_csv(outfile, sep='\t')
           df_fcst_row: pd.DataFrame = df_python[df_python['fcst_lead'] == hr]
           # Set precision to 5 significant figures due to differences in floating point arithmetic
           # in R vs Python
           r_fcst = float(f"{float(r_fcst_row['stat_value']):.4f}")
           df_fcst = float(f"{float(df_fcst_row['stat_value']):.4f}")

           assert (r_fcst == df_fcst)

           # cleanup(outfile)

    except SyntaxError:
        pytest.fail("Error with _calc_stats code")


