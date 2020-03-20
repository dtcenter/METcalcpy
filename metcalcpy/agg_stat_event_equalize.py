import pandas as pd
import argparse
import sys
import yaml
import itertools
import numpy as np

from metcalcpy import event_equalize


class AggStatEventEqualize:
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
        if not self.input_data.empty:
            fix_vals = []
            output_ee_data = pd.DataFrame()
            # list all fixed variables
            if self.params['fixed_vars_vals_input']:
                for value in self.params['fixed_vars_vals_input'].values():
                    fix_vals.append(list(value.values()))

            # permute fix vals
            fix_vals_permuted = list(itertools.chain.from_iterable(fix_vals))
            indy_vals = self.params['indy_vals']

            # perform EE for each forecast variable on y1 axis

            for series_var, series_var_vals in self.params['series_val_1'].items():
                # ungroup series value
                series_var_vals_no_group = []
                for val in series_var_vals:
                    split_val = val.split(',')
                    series_var_vals_no_group.extend(split_val)

                # filter input data based on fcst_var, statistic and all series variables values
                series_data_for_ee = self.input_data[
                    self.input_data[series_var].isin(series_var_vals_no_group)
                ]
                # perform EE on filtered data
                series_data_after_ee = \
                    event_equalize(series_data_for_ee, self.params['indy_var'], indy_vals,
                                   self.params['series_val_1'],
                                   list(self.params['fixed_vars_vals_input'].keys()),
                                   fix_vals_permuted, True, False)

                # append EE data to result
                if output_ee_data.empty:
                    output_ee_data = series_data_after_ee
                else:
                    output_ee_data = output_ee_data.append(series_data_after_ee)

            # if the second Y axis is present - run event equalizer on Y1
            # and then run event equalizer on Y1 and Y2 equalized data

            if self.params['series_val_2']:
                output_ee_data_2 = pd.DataFrame()
                # perform EE for each forecast variable from Y2
                for series_var, series_var_vals in self.params['series_val_2'].items():
                    # ungroup series value
                    series_var_vals_no_group = []
                    for val in series_var_vals:
                        split_val = val.split(',')
                        series_var_vals_no_group.extend(split_val)

                    # filter input data based on fcst_var, statistic
                    # and all series variables values
                    series_data_for_ee = self.input_data[self.input_data[series_var].isin(series_var_vals_no_group)
                    ]
                    # perform EE on filtered data
                    series_data_after_ee = \
                        event_equalize(series_data_for_ee, self.params['indy_var'], indy_vals,
                                       self.params['series_val_2'],
                                       list(self.params['fixed_vars_vals_input'].keys()),
                                       fix_vals_permuted, True,
                                       self.params['line_type'] == "ssvar")

                    # append EE data to result
                    if output_ee_data_2.empty:
                        output_ee_data_2 = series_data_after_ee
                    else:
                        output_ee_data = output_ee_data_2.append(series_data_after_ee)
                output_ee_data = output_ee_data.drop('equalize', axis=1)
                output_ee_data_2 = output_ee_data_2.drop('equalize', axis=1)
                all_ee_records = output_ee_data.append(output_ee_data_2).reindex()
                all_series_vars = {}
                for key in self.params['series_val_2']:
                    all_series_vars[key] = np.unique(self.params['series_val_2'][key]
                                                     + self.params['series_val_2'][key])

                output_ee_data = event_equalize(all_ee_records, self.params['indy_var'], indy_vals,
                                                all_series_vars,
                                                list(self.params['fixed_vars_vals_input'].keys()),
                                                fix_vals_permuted, True,
                                                False)


        else:
            output_ee_data = pd.DataFrame()

        header = True
        mode = 'w'
        export_csv = output_ee_data.to_csv(self.params['agg_stat_output'],
                                           index=None, header=header, mode=mode,
                                           sep="\t", na_rep="NA")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat_event_equalize arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT_EVENT_EQUALIZE = AggStatEventEqualize(PARAMS)
    AGG_STAT_EVENT_EQUALIZE.calculate_values()
