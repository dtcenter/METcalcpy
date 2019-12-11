"""
Program Name: agg_stat.py
"""
import sys
import itertools
import argparse
import yaml
import bootstrapped.bootstrap
from metcalcpy import event_equalize
from metcalcpy.bootstrap_custom import BootstrapDistributionResults, bootstrap_and_value
from metcalcpy.util.ctc_statistics import *
from metcalcpy.util.grad_statistics import *
from metcalcpy.util.sl1l2_statistics import *
from metcalcpy.util.ssvar_statistics import *
from metcalcpy.util.val1l2_statistics import *
from metcalcpy.util.vcnt_statistics import *
from metcalcpy.util.vl1l2_statiatics import *
from metcalcpy.util.ecnt_statistics import *

from metcalcpy.util.utils import is_string_integer, get_derived_curve_name, unique, \
    calc_derived_curve_value, intersection, is_derived_point, parse_bool

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'
__email__ = 'met_help@ucar.edu'


class DerivedCurveComponent:
    """ Holds components and the operation for a derived series
    """

    def __init__(self, first_component, second_component, derived_operation):
        self.first_component = first_component
        self.second_component = second_component
        self.derived_operation = derived_operation


def _sort_data(series_data):
    """ Sorts input data frame by fcst_valid, fcst_lead and stat_name

        Args:
            input pandas data frame
    """
    fields = series_data.keys()
    if "fcst_valid_beg" in fields:
        by_fields = ["fcst_valid_beg", "fcst_lead", "stat_name"]
    elif "fcst_valid" in fields:
        by_fields = ["fcst_valid", "fcst_lead", "stat_name"]
    elif "fcst_init_beg" in fields:
        by_fields = ["fcst_init_beg", "fcst_lead", "stat_name"]
    else:
        by_fields = ["fcst_init", "fcst_lead", "stat_name"]
    series_data = series_data.sort_values(by=by_fields)
    return series_data


class AggStat():
    """A class that performs aggregation statistic logic on input data frame.
           All parameters including data description ann location is in the parameters dictionary
           Usage:
                initialise this call with the parameters dictionary and than
                call calculate_value_and_ci method
                This method will crate and save to the file aggregation statistics
                    agg_stat = AggStat(params)
                    agg_stat.calculate_value_and_ci()
       """

    def __init__(self, in_params):
        """Initialises the class by saving input parameters and reading data from file

             Args:
                 in_params - input parameters as a dictionary
        """

        self.statistic = None
        self.derived_name_to_values = {}
        self.params = in_params

        self.input_data = pd.read_csv(
            self.params['agg_stat_input'],
            header=[0],
            sep='\t'
        )
        self.column_names = self.input_data.columns.values
        self.group_to_value = {}

    EXEMPTED_VARS = ['SSVAR_Spread', 'SSVAR_RMSE']
    STATISTIC_TO_FIELDS = {
        'baser': ['fy_oy', 'fn_oy'],
        'acc': ['fy_oy', 'fn_on'],
        'fbias': ['fy_oy', 'fn_on', 'fy_oy', 'fy_on'],
        'fmean': ['fy_oy', 'fy_on'],
        'pody': ['fy_oy', 'fn_oy'],
        'pofd': ['fy_on', 'fn_on'],
        'podn': ['fn_on', 'fy_on'],
        'far': ['fy_on', 'fy_oy'],
        'csi': ['fy_oy', 'fy_on', 'fn_oy'],
        'gss': ['fy_oy', 'fy_on', 'fn_oy'],
        'hk': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'hss': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'odds': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'lodds': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],
        'baggs': ['fy_oy', 'fn_oy', 'fy_on'],
        'eclv': ['fy_oy', 'fn_oy', 'fy_on', 'fn_on'],

        'fbar': ['fbar'],
        'obar': ['obar'],
        'fstdev': ['fbar', 'ffbar'],
        'ostdev': ['obar', 'oobar'],
        'fobar': ['fobar'],
        'ffbar': ['ffbar'],
        'oobar': ['oobar'],
        'mae': ['mae'],
        'mbias': ['obar', 'fbar'],
        'corr': ['ffbar', 'fbar', 'oobar', 'obar', 'fobar'],
        'anom_corr': ['ffbar', 'fbar', 'oobar', 'obar', 'fobar'],
        'rmsfa': ['ffbar'],
        'rmsoa': ['oobar'],
        'me': ['fbar', 'obar'],
        'me2': ['fbar', 'obar'],
        'mse': ['ffbar', 'oobar', 'fobar'],
        'msess': ['ffbar', 'oobar', 'fobar', 'obar'],
        'rmse': ['ffbar', 'oobar', 'fobar'],
        'estdev': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'bcmse': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'bcrmse': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],

        'fgbar': ['fgbar'],
        'ogbar': ['ogbar'],
        'mgbar': ['mgbar'],
        'egbar': ['egbar'],
        's1': ['egbar', 'mgbar'],
        's1_og': ['egbar', 'ogbar'],
        'fgog_ratio': ['fgbar', 'ogbar'],

        'vcnt_fbar': ['f_speed_bar'],
        'vcnt_obar': ['o_speed_bar'],
        'vcnt_fs_rms': ['uvffbar'],
        'vcnt_os_rms': ['uvoobar'],
        'vcnt_msve': ['uvffbar', 'uvfobar', 'uvoobar'],
        'vcnt_rmsve': ['uvffbar', 'uvfobar', 'uvoobar'],
        'vcnt_fstdev': ['uvffbar', 'f_speed_bar'],
        'vcnt_ostdev': ['uvoobar', 'o_speed_bar'],
        'vcnt_fdir': ['ufbar', 'vfbar'],
        'vcnt_odir': ['uobar', 'vobar'],
        'vcnt_fbar_speed': ['ufbar', 'vfbar'],
        'vcnt_obar_speed': ['uobar', 'vobar'],
        'vcnt_vdiff_speed': ['ufbar', 'uobar', 'vfbar', 'vobar'],
        'vcnt_vdiff_dir': ['ufbar', 'uobar', 'vfbar', 'vobar'],
        'vcnt_speed_err': ['ufbar', 'vfbar', 'uobar', 'vobar'],
        'vcnt_speed_abserr': ['ufbar', 'vfbar', 'uobar', 'vobar'],
        'vcnt_dir_err': ['ufbar', 'vfbar', 'uobar', 'vobar'],
        'vcnt_dir_abser': ['ufbar', 'vfbar', 'uobar', 'vobar'],

        'vl1l2_bias': ['uvffbar', 'uvoobar'],
        'vl1l2_fvar': ['uvffbar', 'f_speed_bar'],
        'vl1l2_ovar': ['uvoobar', 'o_speed_bar'],
        'vl1l2_speed_err': ['ufbar', 'vfbar', 'uobar', 'vobar'],
        'vl1l2_rmsve': ['uvffbar', 'uvfobar', 'uvoobar'],
        'vl1l2_msve': ['uvffbar', 'uvfobar', 'uvoobar'],

        'val1l2_anom_corr':
            ['ufabar', 'vfabar', 'uoabar', 'voabar', 'uvfoabar', 'uvffabar', 'uvooabar'],

        'ssvar_fbar': ['fbar'],
        'ssvar_fstdev': ['fbar', 'ffbar'],
        'ssvar_obar': ['obar'],
        'ssvar_ostdev': ['obar', 'oobar'],
        'ssvar_pr_corr': ['ffbar', 'fbar', 'oobar', 'obar', 'fobar'],
        'ssvar_me': ['fbar', 'obar'],
        'ssvar_estdev': ['obar', 'fbar', 'ffbar', 'oobar', 'fobar'],
        'ssvar_mse': ['ffbar', 'oobar', 'fobar'],
        'ssvar_bcmse': ['fbar', 'obar', 'ffbar', 'oobar', 'fobar'],
        'ssvar_bcrmse': ['fbar', 'obar', 'ffbar', 'oobar', 'fobar'],
        'ssvar_rmse': ['ffbar', 'oobar', 'fobar'],
        'ssvar_anom_corr': ['fbar', 'obar', 'ffbar', 'oobar', 'fobar'],
        'ssvar_me2': ['fbar', 'obar'],
        'ssvar_msess': ['obar', 'oobar', 'ffbar', 'fobar'],
        'ssvar_spread': ['var_mean'],

        'ecnt_crps': ['crps'],
        'ecnt_crpss': ['crps'],
        'ecnt_ign': ['ign'],
        'ecnt_me': ['me'],
        'ecnt_rmse': [],
        'ecnt_spread': ['spread'],
        'ecnt_me_oerr': ['me_oerr'],
        'ecnt_rmse_oerr': [],
        'ecnt_spread_oerr': ['spread_oerr'],
        'ecnt_spread_plus_oerr': ['spread_plus_oerr']
    }

    def _calc_stats(self, values):
        """Calculate the statistic of values for each bootstrap sample
            Args:
                values: a np.array of values we want to calculate the statistic on
                    This is actually a 2d array (matrix) of values. Each row represents
                    a bootstrap resample simulation that we wish to aggregate across.
             Returns:
                a list of calculated statistics

        """
        func_name = 'calculate_{}'.format(self.statistic)
        if values is not None and values.ndim == 2:
            # The single value case
            stat_values = [globals()[func_name](values, self.column_names)]
        elif values is not None and values.ndim == 3:
            # bootstrapped case
            stat_values = []
            for row in values:
                stat_values.append([globals()[func_name](row, self.column_names)])

            # pool = mp.Pool(mp.cpu_count())
            # stat_values = pool.map(partial(globals()['calculate_{}'.format(stat)],
            # columns_names=columns_names), [row for row in data_for_stats])
            # pool.close()
            # pool.join()

        else:
            raise KeyError("can't calculate statistic")
        return stat_values

    def _calc_stats_derived(self, values_both_arrays):
        """Calculate the statistic of values for each derived bootstrap sample
        Args:
            values_both_arrays: a np.array of values we want to calculate the derived statistic on
                This is actually a 2d array (matrix) of values. Each row represents
                a bootstrap resample simulation that we wish to aggregate across.
                The last column contains the derived operation
                The 1st half of columns contains the 1st array data and
                the 2nd - 2nd array data
        Returns:
            a list of calculated derived statistics

        """

        if values_both_arrays is not None and values_both_arrays.ndim == 2:
            # The single value case
            num_of_columns = values_both_arrays.shape[1] - 1
            # get values for the 1st array
            values_1 = values_both_arrays[:, 0:int(num_of_columns / 2)]
            # get values for the 2nd array
            values_2 = values_both_arrays[:, int(num_of_columns / 2):num_of_columns]

            func_name = 'calculate_{}'.format(self.statistic)

            # calculate stat for the 1st array
            stat_values_1 = [globals()[func_name](values_1, self.column_names)]
            # calculate stat for the 2nd array
            stat_values_2 = [globals()[func_name](values_2, self.column_names)]

            # calculate derived stat
            stat_values = calc_derived_curve_value(
                stat_values_1,
                stat_values_2,
                values_both_arrays[0, -1])
        elif values_both_arrays is not None and values_both_arrays.ndim == 3:
            # bootstrapped case
            stat_values = []
            num_of_columns = values_both_arrays.shape[2] - 1
            for row in values_both_arrays:
                # get values for the 1st array
                values_1 = row[:, 0:int(num_of_columns / 2)]
                # get values for the 2nd array
                values_2 = row[:, int(num_of_columns / 2):num_of_columns]

                func_name = 'calculate_{}'.format(self.statistic)

                # calculate stat for the 1st array
                stat_values_1 = [globals()[func_name](values_1, self.column_names)]
                # calculate stat for the 2nd array
                stat_values_2 = [globals()[func_name](values_2, self.column_names)]

                # calculate derived stat
                stat_value = calc_derived_curve_value(
                    stat_values_1,
                    stat_values_2,
                    row[0, -1])
                stat_values.append(stat_value)

            # pool = mp.Pool(mp.cpu_count())
            # stat_values = pool.map(partial(globals()['calculate_{}'.format(stat)],
            # columns_names=columns_names), [row for row in data_for_stats])
            # pool.close()
            # pool.join()

        else:
            raise KeyError("can't calculate statistic")
        return stat_values

    def _prepare_sl1l2_data(self, data_for_prepare):
        """Prepares sl1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_sal1l2_data(self, data_for_prepare):
        """Prepares sal1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_grad_data(self, data_for_prepare):
        """Prepares grad data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_vl1l2_data(self, data_for_prepare):
        """Prepares vl1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_val1l2_data(self, data_for_prepare):
        """Prepares val1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_vcnt_data(self, data_for_prepare):
        """Prepares vcnt data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_ecnt_data(self, data_for_prepare):
        """Prepares ecnt data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        mse = data_for_prepare['rmse'].values * data_for_prepare['rmse'].values
        mse_oerr = data_for_prepare['rmse_oerr'].values * data_for_prepare['rmse_oerr'].values
        crps_climo = data_for_prepare['crps'].values * data_for_prepare['crps'].values

        data_for_prepare['mse'] = mse * data_for_prepare['total'].values
        data_for_prepare['mse_oerr'] = mse_oerr * data_for_prepare['total'].values
        data_for_prepare['crps_climo'] = crps_climo * data_for_prepare['total'].values

        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_ssvar_data(self, data_for_prepare):
        """Prepares ssvar data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

        # rename bin_n column to total
        data_for_prepare.rename(columns={"total": "total_orig", "bin_n": "total"}, inplace=True)
        self.column_names = data_for_prepare.columns.values

        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_ctc_data(self, data_for_prepare):
        """Prepares sl1l2 data.
            Nothing needs to be done

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

    pass

    def _perform_event_equalization(self, indy_vals):
        """ Performs event equalisation on input data
            Args:
             indy_vals: list of independent variable values

        """
        fix_vals = []
        # data frame for the equalised data
        output_ee_data = pd.DataFrame()

        # list all fixed variables
        if self.params['fixed_vars_vals_input']:
            for value in self.params['fixed_vars_vals_input'].values():
                fix_vals.append(list(value.values()))
        # permute fix vals
        fix_vals_permuted = list(itertools.chain.from_iterable(fix_vals))

        # perform EE for each forecast variable
        for fcst_var, fcst_var_stats in self.params['fcst_var_val'].items():
            for fcst_var_stat in fcst_var_stats:
                for series_var, series_var_vals in self.params['series_val'].items():
                    # ungroup series value
                    series_var_vals_no_group = []
                    for val in series_var_vals:
                        split_val = val.split(',')
                        series_var_vals_no_group.extend(split_val)

                    # filter input data based on fcst_var, statistic and all series variables values
                    series_data_for_ee = self.input_data[
                        (self.input_data['fcst_var'] == fcst_var)
                        & (self.input_data["stat_name"] == fcst_var_stat)
                        & (self.input_data[series_var].isin(series_var_vals_no_group))
                        ]
                    # perform EE on filtered data
                    # for SSVAR use equalization of multiple events
                    series_data_after_ee = \
                        event_equalize(series_data_for_ee, self.params['indy_var'], indy_vals,
                                       self.params['series_val'],
                                       list(self.params['fixed_vars_vals_input'].keys()),
                                       fix_vals_permuted, True, self.params['line_type'] == "ssvar")

                    # append EE data to result
                    if output_ee_data.empty:
                        output_ee_data = series_data_after_ee
                    else:
                        output_ee_data.append(series_data_after_ee)
        self.input_data = output_ee_data

    def _get_bootstrapped_stats_for_derived(self, series, distributions):
        """ Calculates aggregation derived statistic value and CI intervals if needed for input data
            Args:
                series: array of lenght =3 where
                1st element - derived series title,
                    ex. 'DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)'
                others  - additional values like indy val and statistic
                distributions - dictionary of the series title
                    to it's BootstrapDistributionResult object

            Returns:
                BootstrapDistributionResults object

        """

        # find all components for the 1st and 2nd series
        derived_curve_component = self.derived_name_to_values[series[0]]
        permute_for_first_series = derived_curve_component.first_component.copy()
        permute_for_first_series.extend(list(series[1:]))
        permute_for_first_series = unique(permute_for_first_series)

        # replace first_series components group names to values
        for i in range(len(permute_for_first_series)):
            if permute_for_first_series[i] in self.group_to_value:
                permute_for_first_series[i] = self.group_to_value[permute_for_first_series[i]]

        permute_for_second_series = derived_curve_component.second_component.copy()
        permute_for_second_series.extend(list(series[1:]))
        permute_for_second_series = unique(permute_for_second_series)

        # replace second_series components group names to values
        for i in range(len(permute_for_second_series)):
            if permute_for_second_series[i] in self.group_to_value:
                permute_for_second_series[i] = self.group_to_value[permute_for_second_series[i]]

        ds_1 = None
        ds_2 = None

        # for each component find  its BootstrapDistributionResult object
        for series_to_distrib_key in distributions.keys():
            if all(elem in permute_for_first_series for elem in series_to_distrib_key):
                ds_1 = distributions[series_to_distrib_key]
            if all(elem in permute_for_second_series for elem in series_to_distrib_key):
                ds_2 = distributions[series_to_distrib_key]
            if ds_1 is not None and ds_2 is not None:
                break

        # if BootstrapDistributionResult object doesn't exist
        # or the original series data size is 0 return an empty object
        if ds_1.values is None or ds_2.values is None \
                or ds_1.values.size == 0 or ds_2.values.size == 0:
            return BootstrapDistributionResults(lower_bound=None,
                                                value=None,
                                                upper_bound=None)

        # validate data
        self._validate_series_cases_for_derived_operation(ds_1.values)
        self._validate_series_cases_for_derived_operation(ds_2.values)

        if self.params['num_iterations'] == 1:
            # don't need bootstrapping and CI calculation -
            # calculate the derived statistic and exit
            stat_val = calc_derived_curve_value(
                [ds_1.value],
                [ds_2.value],
                derived_curve_component.derived_operation)
            results = BootstrapDistributionResults(lower_bound=None,
                                                   value=stat_val[0],
                                                   upper_bound=None)
        else:
            # need bootstrapping and CI calculation in addition to the derived statistic

            # construct joined array with data for series 1 and 2 and operation
            operation = np.full((len(ds_1.values), 1), derived_curve_component.derived_operation)
            values_both_arrays = np.concatenate((ds_1.values, ds_2.values), axis=1)
            values_both_arrays = np.concatenate((values_both_arrays, operation), axis=1)

            try:
                results = bootstrap_and_value(
                    values_both_arrays,
                    stat_func=self._calc_stats_derived,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    alpha=self.params['alpha'],
                    save_data=False)

            except KeyError as err:
                results = bootstrapped.bootstrap.BootstrapResults(None, None, None)
                print(err)
        return results

    def _get_bootstrapped_stats(self, series_data):
        """ Calculates aggregation statistic value and CI intervals if needed for input data
            Args:
                series_data: pandas data frame
            Returns:
                BootstrapDistributionResults object

        """

        # if the data frame is empty - do nothing and return an empty object
        if series_data.empty:
            return BootstrapDistributionResults(lower_bound=None,
                                                value=None,
                                                upper_bound=None)
        # check if derived series are present
        has_derived_series = False
        if self.params['derived_series']:
            has_derived_series = True

        # sort data by dates
        series_data = _sort_data(series_data)

        # find the function that prepares data and execute it
        func = getattr(self, '_prepare_{}_data'.format(self.params['line_type']))
        func(series_data)

        # input data has to be in numpy format for bootstrapping
        data = series_data.to_numpy()

        if self.params['num_iterations'] == 1:
            # don't need bootstrapping and CI calculation -
            # calculate the statistic and exit
            stat_val = self._calc_stats(data)[0]
            results = BootstrapDistributionResults(lower_bound=None,
                                                   value=stat_val,
                                                   upper_bound=None)
            # save original data only if we need it in the future
            # for derived series calculation
            if has_derived_series:
                results.set_original_values(data)
        else:
            # need bootstrapping and CI calculation in addition to statistic
            try:
                results = bootstrap_and_value(
                    data,
                    stat_func=self._calc_stats,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    save_data=has_derived_series)

            except KeyError as err:
                results = BootstrapDistributionResults(None, None, None)
                print(err)
        return results

    def _validate_series_cases_for_derived_operation(self, series_data):
        """ Checks if the derived curve can be calculated.
            The criteria - input array must have only unique
            (fcst_valid, fcst_lead, stat_name) cases.
            Can't calculate differences if  multiple values for one valid date/fcst_lead


            Args:
                series_data: 2d numpu array
            Returns:
                 This method raises an error if this criteria is False
        """
        # find indexes of columns of interests
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
            date_lead_stat = series_data[:, [fcst_valid_ind, fcst_lead_index, stat_name_index]]
            # find the number of unique combinations
            unique_date_size = len(np.vstack({tuple(e) for e in date_lead_stat}))
        except TypeError as err:
            print(err)
            unique_date_size = []

        # identify rows with unique combinations
        ind = np.lexsort(
            (series_data[:, stat_name_index],
             series_data[:, fcst_lead_index], series_data[:, fcst_valid_ind]))
        series_data = series_data[ind, :]

        # the length of the frame with unique combinations should be the same
        # as the number of unique combinations calculated before
        if len(series_data) != unique_date_size \
                and self.params['list_stat'] not in self.EXEMPTED_VARS:
            raise NameError("Derived curve can't be calculated."
                            " Multiple values for one valid date/fcst_lead")

    def _init_out_frame(self, series_fields, series):
        """ Initialises the output frame and add series values to each row
            Args:
                series_fields: list of all possible series fields
                series: list of all series definitions
            Returns:
                pandas data frame
        """
        list_static_val = self.params['list_static_val']
        result = pd.DataFrame()
        row_number = len(series)
        # fill series variables and values
        for static_var in list_static_val:
            result[static_var] = [list_static_val[static_var]] * row_number

        for field_ind, field in enumerate(series_fields):
            result[field] = [row[field_ind] for row in series]

        # fill the stats  and CI values placeholders with None
        result['stat_value'] = [None] * row_number
        result['stat_bcl'] = [None] * row_number
        result['stat_bcu'] = [None] * row_number
        result['nstats'] = [None] * row_number
        return result

    def _get_derived_points(self, series_val, indy_vals):
        """identifies and returns as an list all possible derived points values

            Args:
                series_val: dictionary of series variable to values
                indy_vals: list of independent values
            Returns: a list of all possible values for the each derived points

        """
        series_var = list(series_val.keys())[-1]
        # for each derived series
        for derived_serie in self.params['derived_series']:
            derived_val = series_val.copy()
            derived_val[series_var] = None
            # series 1 components
            ds_1 = derived_serie[0].split(' ')

            # series 2 components
            ds_2 = derived_serie[1].split(' ')

            for var in series_val.keys():
                if derived_val[var] is not None \
                        and intersection(derived_val[var], ds_1) \
                        == intersection(derived_val[var], ds_1):
                    derived_val[var] = intersection(derived_val[var], ds_1)

            derived_curve_name = get_derived_curve_name(derived_serie)
            derived_val[series_var] = [derived_curve_name]
            derived_val[self.params['indy_var']] = indy_vals

            self.derived_name_to_values[derived_curve_name] \
                = DerivedCurveComponent(ds_1, ds_2, derived_serie[-1])
            if ds_1[-1] == ds_2[-1]:
                derived_val['stat_name'] = [ds_1[-1]]
            else:
                derived_val['stat_name'] = [ds_1[-1] + "," + ds_2[-1]]

            return list(itertools.product(*derived_val.values()))

    def calculate_value_and_ci(self):
        """ Calculates aggregated statistics and confidants intervals
            ( if parameter num_iterations > 1) for each series point
            Writes output data to the file

        """
        if not self.input_data.empty:

            # set random seed if present
            if self.params['random_seed'] is not None:
                np.random.seed(self.params['random_seed'])

            is_event_equal = parse_bool(self.params['event_equal'])

            # replace thresh_i values for reliability plot
            indy_vals = self.params['indy_vals']
            if self.params['indy_var'] == 'thresh_i' and self.params['line_type'] == 'pct':
                indy_vals = self.input_data['thresh_i'].sort()
                indy_vals = np.unique(indy_vals)

            # perform groupping
            series_val = self.params['series_val']
            group_to_value_index = 1
            if series_val:
                for key in series_val.keys():
                    for val in series_val[key]:
                        if ',' in val:
                            new_name = 'Group_y1_' + str(group_to_value_index)
                            self.group_to_value[new_name] = val
                            group_to_value_index = group_to_value_index + 1

            # perform EE if needed
            if is_event_equal:
                self._perform_event_equalization(indy_vals)

            if not self.input_data.empty:

                # TODO contourDiff adjustments

                # identify all possible points values by adding series values, indy values
                # and statistics and then permute them
                all_fields_values = series_val.copy()
                all_fields_values[self.params['indy_var']] = indy_vals
                all_fields_values['stat_name'] = self.params['list_stat']
                all_points = list(itertools.product(*all_fields_values.values()))

                if self.params['derived_series']:
                    # identifies and add all possible derived points values
                    all_points.extend(self._get_derived_points(series_val, indy_vals))

                # init the template for output frame
                out_frame = self._init_out_frame(all_fields_values.keys(), all_points)

                point_ind = 0

                # for the each statistic
                for stat_upper in self.params['list_stat']:
                    # save the value to the class variable
                    self.statistic = stat_upper.lower()
                    point_to_distrib = {}
                    # for each point
                    for point in all_points:
                        is_derived = is_derived_point(point)
                        if not is_derived:

                            # filter point data
                            all_filters = []
                            for field_ind, field in enumerate(all_fields_values.keys()):
                                filter_value = point[field_ind]
                                if "," in filter_value:
                                    filter_list = filter_value.split(',')
                                elif ";" in filter_value:
                                    filter_list = filter_value.split(';')
                                else:
                                    filter_list = [filter_value]
                                for i, filter_val in enumerate(filter_list):
                                    if is_string_integer(filter_list[i]):
                                        filter_list[i] = int(filter_list[i])

                                all_filters.append((self.input_data[field].isin(filter_list)))

                            # use numpy to select the rows where any record evaluates to True
                            mask = np.array(all_filters).all(axis=0)
                            point_data = self.input_data.loc[mask]

                            # calculate bootstrap results
                            bootstrap_results = self._get_bootstrapped_stats(point_data)
                            # save bootstrap results
                            point_to_distrib[point] = bootstrap_results
                            n_stats = len(point_data)

                        else:
                            # calculate bootstrap results for the derived point
                            bootstrap_results = self._get_bootstrapped_stats_for_derived(
                                point,
                                point_to_distrib)
                            n_stats = 0

                        # save results to the output data frame
                        out_frame['stat_value'][point_ind] = bootstrap_results.value
                        out_frame['stat_bcl'][point_ind] = bootstrap_results.lower_bound
                        out_frame['stat_bcu'][point_ind] = bootstrap_results.upper_bound
                        out_frame['nstats'][point_ind] = n_stats

                        point_ind = point_ind + 1
            else:
                out_frame = pd.DataFrame()

        else:
            out_frame = pd.DataFrame()

        export_csv = out_frame.to_csv(self.params['agg_stat_output'],
                                      index=None, header=True,
                                      sep="\t", na_rep="NA")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT = AggStat(PARAMS)
    AGG_STAT.calculate_value_and_ci()
