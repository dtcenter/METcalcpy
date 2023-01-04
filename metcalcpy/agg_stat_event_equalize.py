# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: agg_stat_event_equalize.py

How to use:
 - Call from other Python function
        AGG_STAT_EVENT_EQUALIZE = AggStatEventEqualize(PARAMS)
        AGG_STAT_EVENT_EQUALIZE.calculate_values()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python agg_stat_event_equalize.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python agg_stat_event_equalize.eqz.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""
import argparse
import sys
import itertools
import pandas as pd
import yaml
import numpy as np
import warnings

from metcalcpy import GROUP_SEPARATOR
from metcalcpy.event_equalize import event_equalize

class AggStatEventEqualize:
    """A class that performs event equalisation logic on input data
           with MODE and MTD attribute statistics

           All parameters including data description and location is in the parameters dictionary
        Usage:
            initialise this call with the parameters dictionary and then
            calls perform_ee method
            This method will execute EE and save the result to the file
                AGG_STAT_EVENT_EQUALIZE = AggStatEventEqualize(PARAMS)
                AGG_STAT_EVENT_EQUALIZE.calculate_values()
           """

    def __init__(self, in_params):
        self.params = in_params

        self.input_data = pd.read_csv(
            self.params['agg_stat_input'],
            header=[0],
            sep='\t'
        )

        self.column_names = self.input_data.columns.values
        self.series_data = None

    def calculate_values(self):
        """Performs event equalisation if needed and saves equalized data to the file.
        """
        if not self.input_data.empty:

            # list all fixed variables
            if 'fixed_vars_vals_input' in self.params:
                fix_vals_permuted_list = []
                for key in self.params['fixed_vars_vals_input']:
                    vals_permuted = list(itertools.product(*self.params['fixed_vars_vals_input'][key].values()))
                    fix_vals_permuted_list.append(vals_permuted)

                fix_vals_permuted = [item for sublist in fix_vals_permuted_list for item in sublist]

            else:
                fix_vals_permuted = []

            # perform EE for each forecast variable on y1 axis
            output_ee_data = self.run_ee_on_axis(fix_vals_permuted, '1')

            # if the second Y axis is present - run event equalizer on Y1
            # and then run event equalizer on Y1 and Y2 equalized data
            if self.params['series_val_2']:
                output_ee_data_2 = self.run_ee_on_axis(fix_vals_permuted, '2')

                output_ee_data = output_ee_data.drop('equalize', axis=1)
                output_ee_data_2 = output_ee_data_2.drop('equalize', axis=1)
                warnings.simplefilter(action='error', category=FutureWarning)
                all_ee_records = pd.concat([output_ee_data, output_ee_data_2]).reindex()
                all_series_vars = {}
                for key in self.params['series_val_2']:
                    all_series_vars[key] = np.unique(self.params['series_val_2'][key]
                                                     + self.params['series_val_2'][key])

                output_ee_data = event_equalize(all_ee_records, self.params['indy_var'],
                                                all_series_vars,
                                                list(self.params['fixed_vars_vals_input'].keys()),
                                                fix_vals_permuted, True,
                                                False)

        else:
            output_ee_data = pd.DataFrame()

        header = True
        mode = 'w'
        output_ee_data.to_csv(self.params['agg_stat_output'],
                                           index=None, header=header, mode=mode,
                                           sep="\t", na_rep="NA")

    def run_ee_on_axis(self, fix_vals_permuted, axis='1'):
        """Performs event equalisation against previously calculated cases for the selected axis
           Returns:
               A data frame that contains equalized records
       """

        output_ee_data = pd.DataFrame()
        for series_var, series_var_vals in self.params['series_val_' + axis].items():
            # ungroup series value
            series_var_vals_no_group = []
            for val in series_var_vals:
                split_val = val.split(GROUP_SEPARATOR)
                series_var_vals_no_group.extend(split_val)

            # filter input data based on fcst_var, statistic and all series variables values
            series_data_for_ee = self.input_data[
                self.input_data[series_var].isin(series_var_vals_no_group)
            ]
            # perform EE on filtered data
            series_data_after_ee = \
                event_equalize(series_data_for_ee, self.params['indy_var'],
                               self.params['series_val_' + axis],
                               list(self.params['fixed_vars_vals_input'].keys()),
                               fix_vals_permuted, True, False)

            # append EE data to result
            if output_ee_data.empty:
                output_ee_data = series_data_after_ee
            else:
                warnings.simplefilter(action="error", category=FutureWarning)
                output_ee_data = pd.concat([output_ee_data, series_data_after_ee])

        return output_ee_data


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat_event_equalize arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT_EVENT_EQUALIZE = AggStatEventEqualize(PARAMS)
    AGG_STAT_EVENT_EQUALIZE.calculate_values()
