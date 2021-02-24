"""Tests the operation of METcalcpy's event_equalize_against_values code."""

import pandas as pd

from metcalcpy.event_equalize_against_values import event_equalize_against_values


def test_event_equalize_against_values():
    """Tests event equalization against values."""

    indy_var = "fcst_lead"
    series_val = dict({'model': ["AFWAOCv3.5.1_d01", "NoahMPv3.5.1_d01"]})

    fcst_var_val = dict({'APCP_03': ["RATIO_FSA_ASA"]})
    input_data_file = 'data/ee_av_input.data'
    stats_input_data_file = 'data/stats_ee_av_input.data'
    output_data_file = 'data/ee_av_output_py.data'

    # read the input data file into a data frame
    input_data = pd.read_csv(input_data_file, header=[0], sep='\t')
    stats_data = pd.read_csv(stats_input_data_file, header=[0], sep='\t')
    output_data = pd.DataFrame()

    for fcst_var, fcst_var_stats in fcst_var_val.items():
        for series_var, series_var_vals in series_val.items():
            # ungroup series value
            series_var_vals_no_group = []
            for val in series_var_vals:
                split_val = val.split(',')
                series_var_vals_no_group.extend(split_val)

            ee_stats_equalize = input_data[
                (input_data['fcst_var'] == fcst_var)
                & (input_data[series_var].isin(series_var_vals_no_group))
                ]
            f_plot = stats_data[(stats_data['fcst_var'] == fcst_var)
                                & (stats_data[series_var].isin(series_var_vals_no_group))
                                ]
            ee_stats_equalize_unique = (list(set(ee_stats_equalize['equalize'])))
            f_plot = event_equalize_against_values(f_plot, ee_stats_equalize_unique)

            # append EE data to result
            if output_data.empty:
                output_data = f_plot
            else:
                output_data.append(f_plot)
        # save to file
        output_data.to_csv(index=False, sep='\t', path_or_buf=output_data_file)


if __name__ == "__main__":
    test_event_equalize_against_values()
