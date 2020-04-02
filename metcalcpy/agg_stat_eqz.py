import pandas as pd
import argparse
import sys
import yaml
import itertools
import numpy as np

from metcalcpy import event_equalize
from metcalcpy.event_equalize_against_values import event_equalize_against_values
from metcalcpy.util.utils import parse_bool


class AggStatEventEqz:
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
        output_ee_data = self.perform_ee()

        self.input_data = output_ee_data
        header = True
        mode = 'w'
        export_csv = output_ee_data.to_csv(self.params['agg_stat_output'],
                                           index=None, header=header, mode=mode,
                                           sep="\t", na_rep="NA")

    def perform_ee(self):
        is_event_equal = parse_bool(self.params['event_equal'])
        if not self.input_data.empty and is_event_equal:
            ee_stats = pd.read_csv(
                self.params['agg_stat_input_ee'],
                header=[0],
                sep='\t'
            )

            output_ee_data = pd.DataFrame()

            for fcst_var, fcst_var_stats in self.params['fcst_var_val_1'].items():
                for series_var, series_var_vals in self.params['series_val_1'].items():
                    series_var_vals_no_group = []
                    for val in series_var_vals:
                        split_val = val.split(',')
                        series_var_vals_no_group.extend(split_val)

                    # filter input data based on fcst_var, statistic and all series variables values
                    series_data_for_ee = self.input_data[
                        (self.input_data['fcst_var'] == fcst_var)
                        & (self.input_data[series_var].isin(series_var_vals_no_group))
                        ]
                    ee_stats_equalize = ee_stats[
                        (ee_stats['fcst_var'] == fcst_var)
                        & (ee_stats[series_var].isin(series_var_vals_no_group))
                        ]
                    ee_stats_ez_unique = ee_stats_equalize['equalize'].unique()
                    series_data_after_ee = event_equalize_against_values(series_data_for_ee, self.params['indy_var'],
                                                                         ee_stats_ez_unique)
                    # append EE data to result
                    if output_ee_data.empty:
                        output_ee_data = series_data_after_ee
                    else:
                        output_ee_data = output_ee_data.append(series_data_after_ee)

            if self.params['series_val_2']:
                data_axis_2 = pd.DataFrame()
                for fcst_var, fcst_var_stats in self.params['fcst_var_val_2'].items():
                    for series_var, series_var_vals in self.params['series_val_2'].items():
                        series_var_vals_no_group = []
                        for val in series_var_vals:
                            split_val = val.split(',')
                            series_var_vals_no_group.extend(split_val)

                        # filter input data based on fcst_var, statistic and all series variables values
                        series_data_for_ee = self.input_data[
                            (self.input_data['fcst_var'] == fcst_var)
                            & (self.input_data[series_var].isin(series_var_vals_no_group))
                            ]
                        ee_stats_equalize = ee_stats[
                            (ee_stats['fcst_var'] == fcst_var)
                            & (ee_stats[series_var].isin(series_var_vals_no_group))
                            ]
                        ee_stats_ez_unique = ee_stats_equalize['equalize'].unique()

                        series_data_after_ee = event_equalize_against_values(series_data_for_ee,
                                                                             self.params['indy_var'],
                                                                             ee_stats_ez_unique)
                        # append EE data to result
                        if data_axis_2.empty:
                            data_axis_2 = series_data_after_ee
                        else:
                            data_axis_2 = data_axis_2.append(series_data_after_ee)

                # append EE data to result
                if output_ee_data.empty:
                    output_ee_data = data_axis_2
                else:
                    output_ee_data = output_ee_data.append(data_axis_2)

        else:
            output_ee_data = self.input_data
        return output_ee_data


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat_event_eqz arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT_EVENT_EQZ = AggStatEventEqz(PARAMS)
    AGG_STAT_EVENT_EQZ.calculate_values()
