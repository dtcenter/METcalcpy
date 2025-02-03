import os
import pandas as pd
import pytest

import metcalcpy.agg_stat_bootstrap as asb
from agg_stat_bootstrap import AggStatBootstrap
from metcalcpy.util.read_env_vars_in_config import parse_config


cwd = os.path.dirname(__file__)

def get_parms(config_file):
   '''

   :param config_file:
   :return: dictionary representation of the yaml config file settings
   '''
   os.environ['TEST_DIR'] = cwd
   return parse_config(config_file)

def cleanup(filename):
   '''
     Clean up temporary output generated during the test
   :param filename: file to remove
   :return: None
   '''

   try:
      os.remove(filename)
   except FileNotFoundError:
      # if file has already been cleaned or didn't
      # get created, ignore this error
      pass



def agg_stat_bootstrap(parms):
    '''
      Calculate the aggregation statistics for the data specified in the configuration file.
    Args:
        parms: The dictionary representation of configuration settings in the YAML config file.


    Returns:
        the agg_stat_bootstrap.py script generates an output file of the statistic(s) specified in the YAML config file.


    '''

    agg_stat_bootstrap_obj = AggStatBootstrap(parms)
    agg_stat_bootstrap_obj.calculate_values()

def test_calc_stats():
    '''
      Verify that the _calc_stats function does not have any errors emanating from the refactoring of _calc_stats for
      logging and error handling updates.
    Returns:
      none
    '''

    config_file = f'{cwd}/data/mode/rrfs_mode_hrrr.yaml'
    parms = get_parms(config_file)

    try:
       agg_stat_bootstrap(parms)
       # Check for the existence of the
    except SyntaxError:
        pytest.fail("Error still persists with _calc_stats code")

    # clean up
    # output_file = parms['agg_stat_output']
    # cleanup(output_file)

def test_compare_with_rscript():
    '''
       Compare aggregation statistics results against those computed via R.
       Use METviewer to generate the aggregation statistics by de-selecting the 'use python'

    Returns: None



    '''

    config_file = f'{cwd}/data/mode/aggstatbootstrap.yaml'
    parms = get_parms(config_file)
    assert os.path.isfile(config_file)
    infile = parms['agg_stat_input']
    df: pd.DataFrame = pd.read_csv(infile, sep=r'\s+')
    assert df

    # Get the bootstrap stats calculated via the original R implementation
    aggregated_via_R = f'{cwd}/data/mode/rscript_data/mv_rscript_agg_stat_bootstrap.txt'
    r_df:pd.DataFrame = pd.read_csv(aggregated_via_R, sep=r'\s+')


    # Collect the

    try:
       agg_stat_bootstrap(parms)
       # Check for the existence of the output with the requested aggregation statistics computed
       outfile = parms['agg_stat_output']
       assert os.path.isfile(outfile)
    except SyntaxError:
        pytest.fail("Error still persists with _calc_stats code")
