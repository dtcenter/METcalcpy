import os
import pandas as pd
import yaml
import pytest

from metcalcpy.util.utils import  get_met_version

from metcalcpy.agg_stat import AggStat
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
     Clean up temporary agg_stat.py output generated during the test
   :param filename: file to remove
   :return: None
   '''

   try:
      os.remove(filename)
   except FileNotFoundError:
      # if file has already been cleaned or didn't
      # get created, ignore this error
      pass

def aggregate(parms):
   '''
      Calculate all the weighted averages and CI for the columns in the VAL1L2 .txt/.stat file
      from the MET point-stat tool.

      Returns:
         parms: the dictionary representation of the configuration settings in the YAML config file.
         The agg_stat.py script creates an output file.  The name and location are defined in
         the config file.
   '''

   agg_stat_obj = AggStat(parms)
   agg_stat_obj.calculate_stats_and_ci()


@pytest.mark.skip('Not yet updated with new data')
def test_val1l2():
   '''
      Compare MET stat_analysis tool output with
      METcalcpy agg_stat.py result for the DIRA_ME,
      DIRA_MAE, and DIRA_MSE additions to the linetype.

   '''

   # Read in the output generated by the MET stat-analysis tool to retrieve the DIRA_ME, DIRA_MAE,
   # and DIRA_MSE values.
   #
   # Use the MET stat_analysis tool to generate the output:
   # stat_analysis -lookin path-to-MET-val1l2-data -job aggregate -linetype VAL1L2
   # -v 5 -out filename-for-output-file

   # skip the first row of the file, it contains joblist information from stat-analysis
   agg_from_met: pd.DataFrame = pd.read_csv(f"{cwd}/data/stat_analysis/met_val1l2_stat_anal.txt", sep=r'\s+',
                                            skiprows=1)

   # convert all the column names to lower case
   cols = agg_from_met.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in cols]
   agg_from_met.columns = lc_cols
   met_dira_me = float(agg_from_met['dira_me'])
   met_dira_mae = float(agg_from_met['dira_mae'])
   met_dira_mse = float(agg_from_met['dira_mse'])

   # Retrieve the same stat values above from the METcalcpy agg_stat.py output
   # Read in the yaml config file
   config_file = f'{cwd}/val1l2_agg_stat.yaml'

   parms = get_parms(config_file)
   # change the headers of the input data
   # to lower case, the agg_stat.py code is looking for lower case header names
   raw_df: pd.DataFrame = pd.read_csv(parms['agg_stat_input'], sep=r'\s+')
   uc_cols = raw_df.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in uc_cols]
   raw_df.columns = lc_cols
   # create a temporary file with lower case headers, which is what
   # agg_stat.py is expecting
   lc_df_name = "./val1l2.txt"
   raw_df.to_csv(lc_df_name, sep='\t', index=False)
   parms['agg_stat_input'] = lc_df_name
   aggregate(parms)
   calcpy_df = pd.read_csv(parms['agg_stat_output'], sep=r'\s+|\t', engine='python')
   calcpy_dira_me_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VAL1L2_DIRA_ME'])
   calcpy_dira_mae_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VAL1L2_DIRA_MAE'])
   calcpy_dira_mse_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VAL1L2_DIRA_MSE'])
   calcpy_dira_me = float(calcpy_dira_me_df['stat_value'].to_list()[0])
   calcpy_dira_mae = float(calcpy_dira_mae_df['stat_value'].to_list()[0])
   calcpy_dira_mse = float(calcpy_dira_mse_df['stat_value'].to_list()[0])

   calcpy_dira_me_val = round(calcpy_dira_me, 5)
   calcpy_dira_mae_val = round(calcpy_dira_mae, 5)
   calcpy_dira_mse_val = round(calcpy_dira_mse, 5)
   assert calcpy_dira_me_val == met_dira_me
   assert calcpy_dira_mae_val == met_dira_mae
   assert calcpy_dira_mse_val == met_dira_mse

   # clean up
   output_file = parms['agg_stat_output']
   cleanup(output_file)
   cleanup(lc_df_name)

def test_vl1l2():
   '''
      Compare MET stat_analysis tool output with
      METcalcpy agg_stat.py result for the DIR_ME,
      DIR_MAE, and DIR_MSE additions to the linetype.

   '''

   # Read in the output generated by the MET stat-analysis tool to retrieve the DIR_ME, DIR_MAE,
   # and DIR_MSE values.
   #
   # Use the MET stat_analysis tool to generate the output:
   # stat_analysis -lookin path-to-MET-vl1l2-data -job aggregate -linetype VL1L2
   # -v 5 -out filename-for-output-file

   # skip the first row of the file, it contains joblist information from stat-analysis
   agg_from_met: pd.DataFrame = pd.read_csv(f"{cwd}/data/stat_analysis/met_vl1l2_aggregated.txt", sep=r'\s+|\t',
                                            engine='python', skiprows=1)

   # convert all the column names to lower case
   cols = agg_from_met.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in cols]
   agg_from_met.columns = lc_cols
   met_dir_me = float(agg_from_met['dir_me'])
   met_dir_mae = float(agg_from_met['dir_mae'])
   met_dir_mse = float(agg_from_met['dir_mse'])

   # Retrieve the same stat values above from the METcalcpy agg_stat.py output
   # Read in the yaml config file
   config_file = f"{cwd}/vl1l2_agg_stat_met_v12.yaml"

   parms = get_parms(config_file)
   # change the headers of the input data
   # to lower case, the agg_stat.py code is looking for lower case header names
   raw_df: pd.DataFrame = pd.read_csv(parms['agg_stat_input'], sep=r'\s+')
   uc_cols = raw_df.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in uc_cols]
   raw_df.columns = lc_cols
   # create a temporary file with lower case headers, which is what
   # agg_stat.py is expecting
   lc_df_name = "./vl1l2.txt"
   raw_df.to_csv(lc_df_name, sep='\t', index=False)
   parms['agg_stat_input'] = lc_df_name
   # invoking aggregate, now the output dataframe corresponds to the agg_stat_output value
   # defined in the config file
   aggregate(parms)
   calcpy_df = pd.read_csv(parms['agg_stat_output'], sep=r'\s+|\t', engine='python')
   calcpy_dir_me_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VL1L2_DIR_ME'])
   calcpy_dir_mae_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VL1L2_DIR_MAE'])
   calcpy_dir_mse_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VL1L2_DIR_MSE'])
   calcpy_dir_me = float(calcpy_dir_me_df['stat_value'].to_list()[0])
   calcpy_dir_mae = float(calcpy_dir_mae_df['stat_value'].to_list()[0])
   calcpy_dir_mse = float(calcpy_dir_mse_df['stat_value'].to_list()[0])

   calcpy_dir_me_val = round(calcpy_dir_me, 5)
   calcpy_dir_mae_val = round(calcpy_dir_mae, 5)
   calcpy_dir_mse_val = round(calcpy_dir_mse, 5)
   # assert calcpy_dir_me_val == met_dir_me
   assert calcpy_dir_mae_val == met_dir_mae
   assert calcpy_dir_mse_val == met_dir_mse

   # clean up
   output_file = parms['agg_stat_output']
   cleanup(output_file)
   cleanup(lc_df_name)

@pytest.mark.skip('Not yet updated with new data')
def test_vcnt():
   '''
      Compare MET stat_analysis tool output with
      METcalcpy agg_stat.py result for the DIR_ME,
      DIR_MAE,DIR_MSE, and DIR_RMSE additions to the linetype.

   '''

   # Read in the output generated by the MET stat-analysis tool to retrieve the DIR_ME, DIR_MAE,
   # DIR_MSE, and DIR_RMSE values.
   #
   # Use the MET stat_analysis tool to generate the output:
   # stat_analysis -lookin path-to-MET-vl1l2-data -job aggregate_stat -linetype VL1L2
   # -out_line_type VCNT -v 5 -out filename-for-output-file

   # skip the first row of the file, it contains joblist information from stat-analysis
   agg_from_met: pd.DataFrame = pd.read_csv(f"{cwd}/data/stat_analysis/met_vcnt_from_vl1l2_aggstat.txt", sep=r'\s+',
                                            skiprows=1)

   # convert all the column names to lower case
   cols = agg_from_met.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in cols]
   agg_from_met.columns = lc_cols
   met_dir_me = float(agg_from_met['dir_me'])
   met_dir_mae = float(agg_from_met['dir_mae'])
   met_dir_mse = float(agg_from_met['dir_mse'])
   met_dir_rmse = float(agg_from_met['dir_rmse'])

   # Retrieve the same stat values above from the METcalcpy agg_stat.py output
   # Read in the yaml config file
   config_file = f"{cwd}/vcnt_agg_stat.yaml"

   parms = get_parms(config_file)
   # change the headers of the input data
   # to lower case, the agg_stat.py code is looking for lower case header names
   raw_df: pd.DataFrame = pd.read_csv(parms['agg_stat_input'], sep=r'\s+')
   uc_cols = raw_df.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in uc_cols]
   raw_df.columns = lc_cols
   # create a temporary file with lower case headers, which is what
   # agg_stat.py is expecting
   lc_df_name = "./lc_vcnt.txt"
   raw_df.to_csv(lc_df_name, sep='\t', index=False)
   parms['agg_stat_input'] = lc_df_name
   aggregate(parms)
   calcpy_df = pd.read_csv(parms['agg_stat_output'], sep=r'\s+|\t', engine='python')
   calcpy_dir_me_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VCNT_DIR_ME'])
   calcpy_dir_mae_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VCNT_DIR_MAE'])
   calcpy_dir_mse_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VCNT_DIR_MSE'])
   calcpy_dir_rmse_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'VCNT_DIR_RMSE'])
   calcpy_dir_me = float(calcpy_dir_me_df['stat_value'].to_list()[0])
   calcpy_dir_mae = float(calcpy_dir_mae_df['stat_value'].to_list()[0])
   calcpy_dir_mse = float(calcpy_dir_mse_df['stat_value'].to_list()[0])
   calcpy_dir_rmse = float(calcpy_dir_rmse_df['stat_value'].to_list()[0])

   calcpy_dir_me_val = round(calcpy_dir_me, 5)
   calcpy_dir_mae_val = round(calcpy_dir_mae, 5)
   calcpy_dir_mse_val = round(calcpy_dir_mse, 5)
   calcpy_dir_rmse_val = round(calcpy_dir_rmse, 5)
   assert calcpy_dir_me_val == met_dir_me
   assert calcpy_dir_mae_val == met_dir_mae
   assert calcpy_dir_mse_val == met_dir_mse
   assert calcpy_dir_rmse_val == met_dir_rmse

   # clean up
   output_file = parms['agg_stat_output']
   cleanup(output_file)
   cleanup(lc_df_name)

def test_ecnt():
   '''
      Compare MET stat_analysis tool output with
      METcalcpy agg_stat.py result for the IGN_CONV_OERR
      and IGN_CORR_OERR additions to the linetype.

   '''

   # Read in the output generated by the MET stat-analysis tool to retrieve the IGN_CONV_OERR and
   # IGN_CORR_OERR values
   #
   # Use the MET stat_analysis tool to generate the output:
   # stat_analysis -lookin path-to-MET-ecnt-data -job aggregate -linetype ECNT
   # -v 5 -out filename-for-output-file

   # skip the first row of the file, it contains joblist information from stat-analysis
   agg_from_met: pd.DataFrame = pd.read_csv(f"{cwd}/data/stat_analysis/met_ecnt_agg.txt", sep=r'\s+',
                                            skiprows=1)

   # convert all the column names to lower case
   cols = agg_from_met.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in cols]
   agg_from_met.columns = lc_cols
   met_ign_conv_oerr = float(agg_from_met['ign_conv_oerr'])
   met_ign_corr_oerr = float(agg_from_met['ign_corr_oerr'])

   # Retrieve the same stat values above from the METcalcpy agg_stat.py output
   # Read in the yaml config file
   config_file = f"{cwd}/ecnt_agg_stat.yaml"

   parms = get_parms(config_file)
   # change the headers of the input data
   # to lower case, the agg_stat.py code is looking for lower case header names
   raw_df: pd.DataFrame = pd.read_csv(parms['agg_stat_input'], sep=r'\s+')
   uc_cols = raw_df.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in uc_cols]
   raw_df.columns = lc_cols
   # create a temporary file with lower case headers, which is what
   # agg_stat.py is expecting
   lc_df_name = f"{cwd}/lc_ecnt.txt"
   raw_df.to_csv(lc_df_name, sep='\t', index=False)
   parms['agg_stat_input'] = lc_df_name
   aggregate(parms)
   calcpy_df = pd.read_csv(parms['agg_stat_output'], sep=r'\s+|\t', engine='python')
   calcpy_ign_conv_oerr_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'ECNT_IGN_CONV_OERR'])
   calcpy_ign_corr_oerr_df = (calcpy_df.loc[calcpy_df['stat_name'] == 'ECNT_IGN_CORR_OERR'])
   calcpy_ign_conv_oerr = float(calcpy_ign_conv_oerr_df['stat_value'].to_list()[0])
   calcpy_ign_corr_oerr = float(calcpy_ign_corr_oerr_df['stat_value'].to_list()[0])

   calcpy_ign_conv_oerr_val = round(calcpy_ign_conv_oerr, 5)
   calcpy_ign_corr_oerr_val = round(calcpy_ign_corr_oerr, 5)
   assert calcpy_ign_conv_oerr_val == met_ign_conv_oerr
   assert calcpy_ign_corr_oerr_val == met_ign_corr_oerr

   # clean up
   output_file = parms['agg_stat_output']
   cleanup(output_file)
   cleanup(lc_df_name)
