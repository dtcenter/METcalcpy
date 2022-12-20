# ============================*
# ** Copyright UCAR (c) 2022
# ** University Corporation for Atmospheric Research (UCAR)
# ** National Center for Atmospheric Research (NCAR)
# ** Research Applications Lab (RAL)
# ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
# ============================*


"""
Program Name: scorecard.py

How to use:
 - Call from other Python function
        SCORECARD = Scorecard(PARAMS)
        SCORECARD.calculate_scorecard_data()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python scorecard.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python scorecard.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""

from typing import Union
import pandas as pd
import yaml
import logging
import argparse
import sys
import itertools
import re
import statistics
import warnings
import math
import numpy as np
from pandas import DataFrame

from metcalcpy import GROUP_SEPARATOR, DATE_TIME_REGEX
from metcalcpy.util.tost_paired import pt

from metcalcpy.util.utils import intersection, get_derived_curve_name, \
    is_derived_point, is_string_integer, OPERATION_TO_SIGN, calc_derived_curve_value, \
    perfect_score_adjustment, sort_data, PRECISION, DerivedCurveComponent, is_string_strictly_float

COLUMNS_TO_REMOVE = ['equalize', 'stat_ncl', 'stat_ncu', 'stat_bcl', 'stat_bcu', 'fcst_valid_beg', 'fcst_init_beg']


class Scorecard:
    """A class that performs calculation requested statistics for the scorecard. Possible statistic values:
              DIFF_SIG - the difference significance
              DIFF     - mean of difference
              SINGLE   - mean value
           Saves the data into the file
        Usage:
            initialise this call with the parameters dictionary and then
            call calculate_stats method
            This method will crate and save to the file aggregation statistics
                scorecard = Scorecard(params)
                Scorecard.calculate_scorecard_data()
        Raises: EmptyDataError or ValueError when the input DataFrame is empty or doesn't have data
    """

    def __init__(self, in_params):
        """Initialises the class by saving input parameters and reading data from file
                Args:
                    in_params - input parameters as a dictionary
                Raises: EmptyDataError or ValueError when the input DataFrame is empty
                    or doesn't have data
        """

        self.params = in_params
        self.derived_name_to_values = {}
        # import pandas
        try:
            self.input_data = pd.read_csv(
                self.params['sum_stat_input'],
                header=[0],
                sep='\t'
            )

            if self.input_data.empty:
                logging.warning('The input data is empty. The empty output file was created')
                export_csv = self.input_data.to_csv(self.params['sum_stat_output'],
                                                    index=None, header=True, mode="w",
                                                    sep="\t")
                raise ValueError("The input data is empty")

            self.column_names = self.input_data.columns.values
        except pd.errors.EmptyDataError:
            raise
        except KeyError as ex:
            logging.error('Parameter with key %s is missing', ex)
            raise
        self.group_to_value = {}

    def calculate_scorecard_data(self):
        """Calculates requested statistics for the scorecard. Possible statistic values:
              DIFF_SIG - the difference significance
              DIFF     - mean of difference
              SINGLE   - mean value
           Saves the data into the file

        """
        # identify all possible points values by adding series values, indy values
        # and statistics and then permute them
        if self.input_data.empty:
            return pd.DataFrame()
        series_val = self.params['series_val_1']

        all_fields_values = series_val.copy()
        indy_vals = self.params['indy_vals']
        all_fields_values['stat_name'] = self.params['list_stat_1']
        if indy_vals:
            all_fields_values[self.params['indy_var']] = indy_vals
        all_points = list(itertools.product(*all_fields_values.values()))
        if self.params['derived_series_1']:
            # identifies and add all possible derived points values
            all_points.extend(self._get_derived_points(series_val, indy_vals))

        point_to_distrib = {}
        derived_frame = None
        # for each point
        for point_ind, point in enumerate(all_points):
            is_derived = is_derived_point(point)
            if not is_derived:
                # only get the data for each point - no calculations needed
                all_filters = []
                all_filters_pct = []
                filters_wihtout_indy = []
                for field_ind, field in enumerate(all_fields_values.keys()):
                    filter_value = point[field_ind]
                    if GROUP_SEPARATOR in filter_value:
                        filter_list = re.findall(DATE_TIME_REGEX, filter_value)
                        if len(filter_list) == 0:
                            filter_list = filter_value.split(GROUP_SEPARATOR)
                    elif ";" in filter_value:
                        filter_list = filter_value.split(';')
                    else:
                        filter_list = [filter_value]
                    for i, filter_val in enumerate(filter_list):
                        if is_string_integer(filter_val):
                            filter_list[i] = int(filter_val)
                        elif is_string_strictly_float(filter_val):
                            filter_list[i] = float(filter_val)
                    if field in self.input_data.keys():
                        if field != self.params['indy_var']:
                            filters_wihtout_indy.append((self.input_data[field].isin(filter_list)))

                        all_filters.append((self.input_data[field].isin(filter_list)))
                    if field in series_val.keys():
                        all_filters_pct.append((self.input_data[field].isin(filter_list)))

                # use numpy to select the rows where any record evaluates to True
                mask = np.array(all_filters).all(axis=0)
                point_data = self.input_data.loc[mask]
                point_to_distrib[point] = point_data

            else:
                # perform statistics calculation
                results = self._get_stats_for_derived(point, point_to_distrib)
                if results is not None:
                    if derived_frame is None:
                        derived_frame = results
                    else:
                        derived_frame = pd.concat([derived_frame, results], axis=0)

        # print the result to file
        if derived_frame is not None:
            header = True
            mode = 'w'
            if 'append_to_file' in self.params.keys() and self.params['append_to_file'] == 'True':
                header = False
                mode = 'a'
            export_csv = derived_frame.to_csv(self.params['sum_stat_output'],
                                          index=None, header=header, mode=mode,
                                          sep="\t", na_rep="NA", float_format='%.' + str(PRECISION) + 'f')

    def _get_stats_for_derived(self, series, series_to_data) -> Union[DataFrame, None]:
        """ Calculates  derived statistic value for input data
            Args:
                series: array of length = 3 where
                1st element - derived series title,
                    ex. 'DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)'
                others  - additional values like indy val and statistic
                series_to_data - dictionary of the series title to its data as DataFrame

            Returns:
                DataFrame containing 1 row with the resulting data or None

        """

        # get derived name
        derived_name = ''
        for operation in OPERATION_TO_SIGN:
            for point_component in series:
                if point_component.startswith((operation + '(', operation + ' (')):
                    derived_name = point_component
                    break

        # find all components for the 1st and 2nd series
        derived_curve_component = self.derived_name_to_values[derived_name]
        permute_for_first_series = derived_curve_component.first_component.copy()
        for series_comp in series[1:]:
            if series_comp not in permute_for_first_series:
                permute_for_first_series.append(series_comp)

        # replace first_series components group names to values
        for i, perm in enumerate(permute_for_first_series):
            if perm in self.group_to_value:
                permute_for_first_series[i] = self.group_to_value[perm]

        permute_for_second_series = derived_curve_component.second_component.copy()
        for series_comp in series[1:]:
            if series_comp not in permute_for_second_series:
                permute_for_second_series.append(series_comp)

        # replace second_series components group names to values
        for i, perm in enumerate(permute_for_second_series):
            if perm in self.group_to_value:
                permute_for_second_series[i] = self.group_to_value[perm]

        ds_1 = None
        ds_2 = None

        # for each component find  its data as DataFrame
        for series_to_distrib_key in series_to_data.keys():
            if all(elem in permute_for_first_series for elem in series_to_distrib_key):
                ds_1 = series_to_data[series_to_distrib_key]
            if all(elem in permute_for_second_series for elem in series_to_distrib_key):
                ds_2 = series_to_data[series_to_distrib_key]
            if ds_1 is not None and ds_2 is not None:
                break

        if ds_1.values is None or ds_2.values is None \
                or ds_1.values.size == 0 or ds_2.values.size == 0:
            return None

        # validate data
        if derived_curve_component.derived_operation != 'SINGLE' \
                and len(ds_1.values) != 0 and len(ds_2.values):
            fcst_lead_index = np.where(self.column_names == 'fcst_lead')[0][0]
            stat_name_index = np.where(self.column_names == 'stat_name')[0][0]
            if "fcst_valid_beg" in self.column_names:
                fcst_valid_ind = np.where(self.column_names == 'fcst_valid_beg')[0][0]
            elif "fcst_valid" in self.column_names:
                fcst_valid_ind = np.where(self.column_names == 'fcst_valid')[0][0]
            elif "fcst_init_beg" in self.column_names:
                fcst_valid_ind = \
                    np.where(self.column_names == 'fcst_init_beg')[0][0]
            else:
                fcst_valid_ind = \
                    np.where(self.column_names == 'fcst_init')[0][0]
            try:
                # filter columns of interest
                date_lead_stat = ds_1.values[:, [fcst_valid_ind, fcst_lead_index, stat_name_index]]
                # find the number of unique combinations
                unique_date_size = len(set(map(tuple, date_lead_stat)))
            except TypeError as err:
                print(err)
                unique_date_size = []
            if unique_date_size != len(ds_1.values):
                raise NameError("Derived curve can't be calculated."
                                " Multiple values for one valid date/fcst_lead")

        # sort data by dates
        ds_1_values = sort_data(ds_1)
        ds_2_values = sort_data(ds_2)

        stat_values_1 = ds_1_values['stat_value'].tolist()
        stat_values_2 = ds_2_values['stat_value'].tolist()

        # calculate derived statistic based on the operation and stat_flag
        derived_stat = None
        if derived_curve_component.derived_operation == 'DIFF_SIG':
            if self.params['stat_flag'] == 'EMC':
                derived_stat = self._calculate_diff_sig_emc(stat_values_1, stat_values_2)
            else:
                derived_stat = self._calculate_diff_sig_ncar(stat_values_1, stat_values_2)

        elif derived_curve_component.derived_operation == 'DIFF':
            # perform the derived operation
            derived_stat_list = calc_derived_curve_value(
                stat_values_1,
                stat_values_2,
                'DIFF')
            derived_stat = statistics.mean(derived_stat_list)

        elif derived_curve_component.derived_operation == 'SINGLE':
            derived_stat = statistics.mean(stat_values_1)

        # create dataframe from teh 1st row of the original data
        df = ds_1.head(1)
        # remove unneeded columns
        for column in COLUMNS_TO_REMOVE:
            if column in self.column_names:
                df = df.drop(column, axis=1)
        df.reset_index(drop=True, inplace=True)

        # set statistic value
        df.at[0, 'stat_value'] = derived_stat
        df = df.astype({"fcst_lead": str})

        # set fcst_lead
        for fcst_lead in self.params["series_val_1"]['fcst_lead']:
            for component in derived_curve_component.first_component:
                if component == fcst_lead:
                    with warnings.catch_warnings():
                        warnings.simplefilter(action="ignore", category=Warning)
                        df.at[0, 'fcst_lead'] = fcst_lead
                    break

        # set model
        df.at[0, 'model'] = derived_name
        return df

    def _calculate_diff_sig_ncar(self, ds_1_values, ds_2_values) -> float:
        """ calculates the difference significance between two data arrays using p-value

            Args:
                    ds_1_values: list of floats
                    ds_2_values: list of floats
            Returns: the difference significance

        """
        # perform the derived operation
        derived_stat_list = calc_derived_curve_value(ds_1_values, ds_2_values, 'DIFF_SIG')
        avg = statistics.mean(derived_stat_list)
        sdv = statistics.stdev(derived_stat_list)
        total = len(derived_stat_list)
        t = avg / (sdv / np.sqrt(total))
        p_val = 1 - 2 * pt(abs(t), total - 1, lower_tail=False)
        derived_stat = perfect_score_adjustment(statistics.mean(ds_1_values),
                                                statistics.mean(ds_2_values),
                                                self.params['list_stat_1'][0],
                                                p_val)
        return derived_stat

    def _calculate_diff_sig_emc(self, ds_1_values, ds_2_values) -> float:
        """ calculates the difference significance between two data arrays using EMC method

            Args:
                    ds_1_values: list of floats
                    ds_2_values: list of floats
            Returns: the difference significance

        """
        derived_stat = None
        values_1 = np.array(ds_1_values)
        values_2 = np.array(ds_2_values)
        val2_minus_val1 = np.subtract(values_2, values_1)
        acdm = sum(val2_minus_val1) / self.params['ndays']
        acdm_list = [acdm] * len(values_1)
        std = math.sqrt(sum(np.subtract(val2_minus_val1, acdm_list) * np.subtract(val2_minus_val1, acdm_list)) /
                        self.params['ndays'])
        nsz = len(ds_1_values)
        intvl = round(1.960 * std / math.sqrt(nsz - 1), 6)
        mean1 = round(statistics.mean(values_1), 6)
        mean2 = round(statistics.mean(values_2), 6)
        if self.params['list_stat_1'][0].startswith('ME') or self.params['list_stat_1'][0].startswith('BIAS'):
            ds = (abs(mean2 - mean1)) / intvl
            sss = abs(mean2) - abs(mean1)
            if sss is not None and sss < 0:
                ds = (-1) * ds
        elif self.params['list_stat_1'][0].startswith('RMSE') \
                or self.params['list_stat_1'][0].startswith('RMSVE'):
            ds = (mean2 - mean1) / intvl
        else:
            ds = (mean1 - mean2) / intvl
        if ds is not None:
            ds = round(ds, 3)
            if self.params['ndays'] >= 80:
                alpha1 = 1.960  # 95% confidence level
                alpha2 = 2.576  # 99% confidence level
                alpha3 = 3.291  # 99.9% confidence level
            elif self.params['ndays'] >= 40 and self.params['ndays'] < 80:
                alpha1 = 2.0  # 95% confidence level
                alpha2 = 2.66  # 99% confidence level
                alpha3 = 3.46  # 99.9% confidence level
            elif self.params['ndays'] >= 20 and self.params['ndays'] < 40:
                alpha1 = 2.042  # 95% confidence level
                alpha2 = 2.75  # 99% confidence level
                alpha3 = 3.646  # 99.9% confidence level
            elif self.params['ndays'] < 20:
                alpha1 = 2.228  # 95% confidence level
                alpha2 = 3.169  # 99% confidence level
                alpha3 = 4.587  # 99.9% confidence level
            ds95 = ds
            ds99 = ds * alpha1 / alpha2
            ds99 = round(ds99, 3)
            ds999 = ds * alpha1 / alpha3;
            ds999 = round(ds999, 3)
            if ds999 >= 1:
                derived_stat = 1
            elif ds999 < 1 and ds99 >= 1:
                derived_stat = 0.99
            elif ds99 < 1 and ds95 >= 1:
                derived_stat = 0.95
            elif ds95 > -1 and ds95 < 1:
                derived_stat = 0
            elif ds95 <= -1 and ds99 > -1:
                derived_stat = -0.95
            elif ds99 <= -1 and ds999 > -1:
                derived_stat = -0.99
            elif ds999 <= -1 and ds999 > -100.0:
                derived_stat = -1
            elif ds999 < -100.0:
                derived_stat = -1
        return derived_stat

    def _get_derived_points(self, series_val, indy_vals):
        """identifies and returns as a list all possible derived points values

            Args:
                series_val: dictionary of series variable to values
                indy_vals: list of independent values
            Returns: a list of all possible values for each derived points

        """

        # for each derived series
        result = []
        for derived_serie in self.params['derived_series_1']:
            # series 1 components
            ds_1 = derived_serie[0].split(' ')

            # series 2 components
            ds_2 = derived_serie[1].split(' ')
            # find a variable of the operation by comparing values in each derived series component
            series_var_vals = ()
            for ind, name in enumerate(ds_1):
                if name != ds_2[ind]:
                    series_var_vals = (name, ds_2[ind])
                    break

            series_var = list(series_val.keys())[-1]
            if len(series_var_vals) > 0:
                for var in series_val.keys():
                    if all(elem in series_val[var] for elem in series_var_vals):
                        series_var = var
                        break

            derived_val = series_val.copy()
            derived_val[series_var] = None

            for var in series_val.keys():
                if derived_val[var] is not None \
                        and intersection(derived_val[var], ds_1) \
                        == intersection(derived_val[var], ds_1):
                    derived_val[var] = intersection(derived_val[var], ds_1)

            derived_curve_name = get_derived_curve_name(derived_serie)
            derived_val[series_var] = [derived_curve_name]
            if len(indy_vals) > 0:
                derived_val[self.params['indy_var']] = indy_vals

            self.derived_name_to_values[derived_curve_name] \
                = DerivedCurveComponent(ds_1, ds_2, derived_serie[-1])
            if ds_1[-1] == ds_2[-1]:
                derived_val['stat_name'] = [ds_1[-1]]
            else:
                derived_val['stat_name'] = [ds_1[-1] + "," + ds_2[-1]]
            result.append(list(itertools.product(*derived_val.values())))

        return [y for x in result for y in x]


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of scorecard arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    SCORECARD = Scorecard(PARAMS)
    SCORECARD.calculate_scorecard_data()
