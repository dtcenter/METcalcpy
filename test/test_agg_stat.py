import pytest
import pandas as pd
import yaml
from metcalcpy.agg_stat import AggStat


def get_parms(config_file):
   '''

   :param config_file:
   :return: dictionary representation of the yaml config file settings
   '''

   with open(config_file, 'r') as stream:
      try:
         parms: dict = yaml.load(stream, Loader=yaml.FullLoader)
      except yaml.YAMLError as exc:
         print(exc)
   return parms


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



def test_val1l2():
   ''' 
      Compare MET stat_analysis tool output with 
      METcalcpy agg_stat.py result for the DIRA_ME,
      DIRA_MAE, and DIRA_MSE additions to the linetype.
      
   '''


   # Read in the output generated by the MET stat-analysis tool to retrieve the DIR_ME, DIR_MAE,
   # and DIR_MSE values.
   #
   # Use the MET stat_analysis tool to generate the output:
   # stat_analysis -lookin path-to-MET-val1l2-data -job aggregate -linetype VAL1L2
   # -v 5 -out filename-for-output-file



   # skip the first row of the file, it contains joblist information from stat-analysis
   agg_from_met: pd.DataFrame = pd.read_csv("./data/stat_analysis/met_val1l2_stat_anal.txt", sep='\s+',
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
   config_file = './val1l2_agg_stat.yaml'

   parms = get_parms(config_file)
   # change the headers of the input data
   # to lower case, the agg_stat.py code is looking for lower case header names
   raw_df: pd.DataFrame = pd.read_csv(parms['agg_stat_input'], sep='\s+')
   uc_cols = raw_df.columns.to_list()
   lc_cols = [lc_cols.lower() for lc_cols in uc_cols]
   raw_df.columns = lc_cols
   # create a temporary file with lower case headers, which is what
   # agg_stat.py is expecting
   lc_df_name = "./lc_val1l2.txt"
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

