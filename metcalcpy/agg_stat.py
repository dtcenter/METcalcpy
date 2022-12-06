# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: agg_stat.py

How to use:
 - Call from other Python function
        AGG_STAT = AggStat(PARAMS)
        AGG_STAT.calculate_stats_and_ci()
        where PARAMS – a dictionary with data description parameters including
        location of input and output data.
        The structure is similar to Rscript template

 - Run as a stand-alone script
        python agg_stat.py <parameters_file>
        where - <parameters_file> is YAML file with parameters
        and environment variable should be set to PYTHONPATH=<path_to_METcalcpy>

 - Run from Java
        proc = Runtime.getRuntime().exec(
                “python agg_stat.py <parameters_file>”,
                new String[]{”PYTHONPATH=<path_to_METcalcpy>”},
                new File(System.getProperty("user.home")));

"""
import sys
import itertools
import argparse
from inspect import signature
import yaml
import pandas

from metcalcpy import GROUP_SEPARATOR, DATE_TIME_REGEX
from metcalcpy.bootstrap import bootstrap_and_value, BootstrapResults
from metcalcpy.util.ctc_statistics import *
from metcalcpy.util.grad_statistics import *
from metcalcpy.util.sl1l2_statistics import *
from metcalcpy.util.sal1l2_statistics import *
from metcalcpy.util.ssvar_statistics import *
from metcalcpy.util.val1l2_statistics import *
from metcalcpy.util.vcnt_statistics import *
from metcalcpy.util.vl1l2_statistics import *
from metcalcpy.util.ecnt_statistics import *
from metcalcpy.util.nbrcnt_statistics import *
from metcalcpy.util.nbrctc_statistics import *
from metcalcpy.util.pstd_statistics import *
from metcalcpy.util.rps_statistics import *
from metcalcpy.util.mcts_statistics import *

from metcalcpy.util.utils import is_string_integer, get_derived_curve_name, \
    calc_derived_curve_value, intersection, is_derived_point, parse_bool, \
    OPERATION_TO_SIGN, perfect_score_adjustment, perform_event_equalization, \
    aggregate_field_values, sort_data, DerivedCurveComponent, is_string_strictly_float

__author__ = 'Tatiana Burek'


class AggStat:
    """A class that performs aggregation statistic logic on input data frame.
           All parameters including data description and location is in the parameters dictionary
           Usage:
                initialise this call with the parameters dictionary and than
                call calculate_stats_and_ci method
                This method will crate and save to the file aggregation statistics
                    agg_stat = AggStat(params)
                    agg_stat.calculate_stats_and_ci()
            Raises: EmptyDataError or ValueError when the input DataFrame is empty
                or doesn't have data
       """

    def __init__(self, in_params):
        """Initialises the class by saving input parameters and reading data from file

            Args:
                in_params - input parameters as a dictionary
            Raises: EmptyDataError or ValueError when the input DataFrame is empty
                or doesn't have data
        """

        self.statistic = None
        self.derived_name_to_values = {}
        self.params = in_params

        try:
            self.input_data = pd.read_csv(
                self.params['agg_stat_input'],
                header=[0],
                sep='\t'
            )
            self.column_names = self.input_data.columns.values
        except pandas.errors.EmptyDataError:
            raise
        except KeyError as er:
            print(f'ERROR: parameter with key {er} is missing')
            raise
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
        'anom_corr': ['ffabar', 'fabar', 'ooabar', 'oabar', 'foabar'],
        'anom_corr_raw': ['ffabar', 'ooabar', 'foabar'],
        'rmsfa': ['ffabar'],
        'rmsoa': ['ooabar'],
        'me': ['fbar', 'obar'],
        'me2': ['fbar', 'obar'],
        'mse': ['ffbar', 'oobar', 'fobar'],
        'msess': ['ffbar', 'oobar', 'fobar', 'obar'],
        'rmse': ['ffbar', 'oobar', 'fobar'],
        'si': ['ffbar', 'oobar', 'fobar', 'obar'],
        'estdev': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'bcmse': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'bcrmse': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],
        'pr_corr': ['ffbar', 'oobar', 'fobar', 'fbar', 'obar'],

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
        'vcnt_anom_corr': ['uvffabar', 'uvfoabar', 'uvooabar', 'fa_speed_bar', 'oa_speed_bar'],
        'vcnt_anom_corr_uncntr': ['uvffabar', 'uvfoabar', 'uvooabar'],

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
        'ecnt_crpss': ['crps', 'crpscl'],
        'ecnt_ign': ['ign'],
        'ecnt_me': ['me'],
        'ecnt_rmse': [],
        'ecnt_spread': ['spread'],
        'ecnt_me_oerr': ['me_oerr'],
        'ecnt_rmse_oerr': [],
        'ecnt_spread_oerr': ['spread_oerr'],
        'ecnt_spread_plus_oerr': ['spread_plus_oerr'],
        'ecnt_crpscl': ['crpscl'],
        'ecnt_crps_emp': ['crps_emp'],
        'ecnt_crpscl_emp': ['crpscl_emp'],
        'ecnt_crpss_emp': ['crpscl_emp', 'crps_emp'],
        'ecnt_crps_emp_fair': ['crps_emp_fair'],
        'ecnt_spread_md': ['spread_md'],
        'ecnt_mae': ['mae'],
        'ecnt_mae_oerr': ['mae_oerr'],
        'ecnt_n_ge_obs': [],
        'ecnt_me_ge_obs': ['me_ge_obs'],
        'ecnt_n_lt_obs': [],
        'ecnt_me_lt_obs': ['me_lt_obs'],
        'ecnt_bias_ratio': ['me_ge_obs','me_lt_obs'],

        'nbr_fbs': ['fbs'],
        'nbr_fss': ['fss'],
        'nbr_afss': ['afss'],
        'nbr_f_rare': ['f_rate'],
        'nbr_o_rare': ['o_rate'],

    }

    def _calc_stats(self, values):
        """Calculate the statistic of values for each bootstrap sample
            Args:
                values: a np.array of values we want to calculate the statistic on
                    This is actually a 2d array (matrix) of values. Each row represents
                    a bootstrap resample simulation that we wish to aggregate across.
             Returns:
                a list of calculated statistics
            Raises:
                an error

        """
        func_name = f'calculate_{self.statistic}'
        # some functions have an extra 3rd parameter that represents
        # if some data preliminary data aggregation was done
        # if this parameter is present we need to add it
        num_parameters = len(signature(globals()[func_name]).parameters)

        if values is not None and values.ndim == 2:

            # The single value case
            if num_parameters == 2:
                stat_values = [globals()[func_name](values, self.column_names)]
            else:
                stat_values = [globals()[func_name](values, self.column_names, True)]
        elif values is not None and values.ndim == 3:
            # bootstrapped case
            stat_values = []
            for row in values:
                if num_parameters == 2:
                    stat_value = [globals()[func_name](row, self.column_names)]
                else:
                    stat_value = [globals()[func_name](row, self.column_names, True)]

                stat_values.append(stat_value)

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

            func_name = f'calculate_{self.statistic}'
            # some functions have an extra 3rd parameter that represents
            # if some data preliminary data aggregation was done
            # if this parameter is present we need to add it
            num_parameters = len(signature(globals()[func_name]).parameters)

            if num_parameters == 2:
                # calculate stat for the 1st array
                stat_values_1 = [globals()[func_name](values_1, self.column_names)]
                # calculate stat for the 2nd array
                stat_values_2 = [globals()[func_name](values_2, self.column_names)]
            else:
                # calculate stat for the 1st array
                stat_values_1 = [globals()[func_name](values_1, self.column_names, True)]
                # calculate stat for the 2nd array
                stat_values_2 = [globals()[func_name](values_2, self.column_names, True)]

            # calculate derived stat
            stat_values = calc_derived_curve_value(
                stat_values_1,
                stat_values_2,
                values_both_arrays[0, -1])
            if not isinstance(stat_values, list):
                stat_values = [stat_values]
        elif values_both_arrays is not None and values_both_arrays.ndim == 3:
            # bootstrapped case
            stat_values = []
            num_of_columns = values_both_arrays.shape[2] - 1
            for row in values_both_arrays:

                # get values for the 1st array
                values_1 = row[:, 0:int(num_of_columns / 2)]
                # get values for the 2nd array
                values_2 = row[:, int(num_of_columns / 2):num_of_columns]

                func_name = f'calculate_{self.statistic}'
                # some functions have an extra 3rd parameter that represents
                # if some data preliminary data aggregation was done
                # if this parameter is present we need to add it
                num_parameters = len(signature(globals()[func_name]).parameters)

                if num_parameters == 2:
                    # calculate stat for the 1st array
                    stat_values_1 = [globals()[func_name](values_1, self.column_names)]
                    # calculate stat for the 2nd array
                    stat_values_2 = [globals()[func_name](values_2, self.column_names)]
                else:
                    # calculate stat for the 1st array
                    stat_values_1 = [globals()[func_name](values_1, self.column_names, True)]
                    # calculate stat for the 2nd array
                    stat_values_2 = [globals()[func_name](values_2, self.column_names, True)]

                # calculate derived stat
                stat_value = calc_derived_curve_value(
                    stat_values_1,
                    stat_values_2,
                    row[0, -1])
                if not isinstance(stat_value, list):
                    stat_value = [stat_value]
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
        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                data_for_prepare[column] \
                    = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_sal1l2_data(self, data_for_prepare):
        """Prepares sal1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                data_for_prepare[column] \
                    = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_grad_data(self, data_for_prepare):
        """Prepares grad data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                data_for_prepare[column] \
                    = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_vl1l2_data(self, data_for_prepare):
        """Prepares vl1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                data_for_prepare[column] \
                    = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_val1l2_data(self, data_for_prepare):
        """Prepares val1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                data_for_prepare[column] \
                    = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_vcnt_data(self, data_for_prepare):
        """Prepares vcnt data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
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
        # crps_climo = data_for_prepare['crps'].values * data_for_prepare['crps'].values

        variance = data_for_prepare['spread'].values * data_for_prepare['spread'].values
        variance_oerr = data_for_prepare['spread_oerr'].values * data_for_prepare['spread_oerr'].values
        variance_plus_oerr = data_for_prepare['spread_oerr'].values * data_for_prepare['spread_oerr'].values

        data_for_prepare['mse'] = mse * data_for_prepare['total'].values
        data_for_prepare['mse_oerr'] = mse_oerr * data_for_prepare['total'].values
        # data_for_prepare['crps_climo'] = crps_climo * data_for_prepare['total'].values

        data_for_prepare['variance'] = variance * data_for_prepare['total'].values
        data_for_prepare['variance_oerr'] = variance_oerr * data_for_prepare['total'].values
        data_for_prepare['variance_plus_oerr'] = variance_plus_oerr * data_for_prepare['total'].values

        self.column_names = data_for_prepare.columns.values

        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                if column == 'me_ge_obs':
                    data_for_prepare[column] \
                        = data_for_prepare[column].values * data_for_prepare['n_ge_obs'].values
                elif column == 'me_lt_obs':
                    data_for_prepare[column] \
                        = data_for_prepare[column].values * data_for_prepare['n_lt_obs'].values
                else:
                    data_for_prepare[column] \
                        = data_for_prepare[column].values * data_for_prepare['total'].values

    def _prepare_rps_data(self, data_for_prepare):
        total = data_for_prepare['total'].values
        d_rps_climo = data_for_prepare['rps'].values / (1 - data_for_prepare['rpss'].values)
        data_for_prepare['rps_climo'] = d_rps_climo * total
        data_for_prepare['rps'] = data_for_prepare['rps'].values * total
        data_for_prepare['rps_comp'] = data_for_prepare['rps_comp'].values * total
        self.column_names = data_for_prepare.columns.values

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

    def _prepare_nbr_cnt_data(self, data_for_prepare):
        """Prepares nbrcnt data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

        total = data_for_prepare['total'].values
        fbs = total * data_for_prepare['fbs'].values
        fss_den = (data_for_prepare['fbs'].values / (1.0 - data_for_prepare['fss'].values)) * total

        f_rate = total * data_for_prepare['f_rate'].values
        data_for_prepare['fbs'] = fbs
        data_for_prepare['fss'] = fss_den
        data_for_prepare['f_rate'] = f_rate

    def _prepare_pct_data(self, data_for_prepare):
        """Prepares pct data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

    def _prepare_mctc_data(self, data_for_prepare):
        """Prepares mctc data.
           Nothing needs to be done

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        if 'ec_value' in data_for_prepare.columns:
            if not (data_for_prepare['ec_value'] == data_for_prepare['ec_value'][0]).all():
                raise ValueError('EC_VALUE is NOT constant across  MCTC lines')

    def _prepare_ctc_data(self, data_for_prepare):
        """Prepares CTC data.
            Checks if all values from ec_value column are the same and if not - throws an error

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

        if 'ec_value' in data_for_prepare.columns:
            if not (data_for_prepare['ec_value'] == data_for_prepare['ec_value'][0]).all():
                raise ValueError('EC_VALUE is NOT constant across  CTC lines')


    def _prepare_cts_data(self, data_for_prepare):
        """Prepares cts data.
            Checks if all values from ec_value column are the same and if not - throws an error

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

        if 'ec_value' in data_for_prepare.columns:
            if not (data_for_prepare['ec_value'] == data_for_prepare['ec_value'][0]).all():
                raise ValueError('EC_VALUE is NOT constant across  CTS lines')

    def _prepare_nbr_ctc_data(self, data_for_prepare):
        """Prepares MBR_CTC data.
            Nothing needs to be done

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """

    def _get_bootstrapped_stats_for_derived(self, series, distributions, axis="1"):
        """ Calculates aggregation derived statistic value and CI intervals if needed for input data
            Args:
                series: array of length = 3 where
                1st element - derived series title,
                    ex. 'DIFF(ENS001v3.6.1_d01 DPT FBAR-ENS001v3.6.1_d02 DPT FBAR)'
                others  - additional values like indy val and statistic
                distributions - dictionary of the series title
                    to it's BootstrapDistributionResult object

            Returns:
                BootstrapDistributionResults object

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
            return BootstrapResults(lower_bound=None,
                                    value=None,
                                    upper_bound=None)
        # calculate the number of values in the group if the series has a group
        # it is need d for the validation
        num_diff_vals_first = 0
        num_diff_vals_second = 0
        for val in permute_for_first_series:
            size = len(val.split(GROUP_SEPARATOR))
            if size > 1:
                num_diff_vals_first = num_diff_vals_first + size
        for val in permute_for_second_series:
            size = len(val.split(GROUP_SEPARATOR))
            if size > 1:
                num_diff_vals_second = num_diff_vals_second + size
        if num_diff_vals_first == 0:
            num_diff_vals_first = 1
        if num_diff_vals_second == 0:
            num_diff_vals_second = 1

        # validate data
        if derived_curve_component.derived_operation != 'SINGLE':
            self._validate_series_cases_for_derived_operation(ds_1.values, axis, num_diff_vals_first)
            self._validate_series_cases_for_derived_operation(ds_2.values, axis, num_diff_vals_second)

        if self.params['num_iterations'] == 1 or derived_curve_component.derived_operation == 'ETB':
            # don't need bootstrapping and CI calculation -
            # calculate the derived statistic and exit

            if derived_curve_component.derived_operation == 'ETB':
                index_array = np.where(self.column_names == 'stat_value')[0]
                func_name = f'calculate_{self.statistic}'
                for row in ds_1.values:
                    stat = [globals()[func_name](row[np.newaxis, ...], self.column_names)]
                    row[index_array] = stat
                for row in ds_2.values:
                    stat = [globals()[func_name](row[np.newaxis, ...], self.column_names)]
                    row[index_array] = stat

                ds_1_value = ds_1.values[:, index_array].flatten().tolist()
                ds_2_value = ds_2.values[:, index_array].flatten().tolist()
            else:
                ds_1_value = [ds_1.value]
                ds_2_value = [ds_2.value]

            stat_val = calc_derived_curve_value(
                ds_1_value,
                ds_2_value,
                derived_curve_component.derived_operation)
            if stat_val is not None:
                results = BootstrapResults(lower_bound=None,
                                       value=round_half_up(stat_val[0], 5),
                                       upper_bound=None)
            else:
                results = BootstrapResults(lower_bound=None,
                                           value=None,
                                           upper_bound=None)
            results.set_distributions([results.value])
        else:
            # need bootstrapping and CI calculation in addition to the derived statistic

            # construct joined array with data for series 1 and 2 and operation
            operation = np.full((len(ds_1.values), 1), derived_curve_component.derived_operation)
            values_both_arrays = np.concatenate((ds_1.values, ds_2.values), axis=1)
            values_both_arrays = np.concatenate((values_both_arrays, operation), axis=1)

            try:
                # calculate a block length for the circular temporal block bootstrap if needed
                block_length = 1

                # to use circular block bootstrap or not
                is_cbb = True
                if 'circular_block_bootstrap' in self.params.keys():
                    is_cbb = parse_bool(self.params['circular_block_bootstrap'])

                if is_cbb:
                    block_length = int(math.sqrt(len(values_both_arrays)))
                results = bootstrap_and_value(
                    values_both_arrays,
                    stat_func=self._calc_stats_derived,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    alpha=self.params['alpha'],
                    save_data=False,
                    save_distributions=derived_curve_component.derived_operation == 'DIFF_SIG',
                    block_length=block_length)
            except KeyError as err:
                results = BootstrapResults(None, None, None)
                print(err)

        if derived_curve_component.derived_operation == 'DIFF_SIG':
            # remove None values in distributions
            distributions = [i for i in results.distributions if i is not None]
            diff_sig = None
            if distributions and results.value is not None:
                distribution_mean = np.mean(distributions)
                distribution_under_h0 = distributions - distribution_mean
                pval = np.mean(np.absolute(distribution_under_h0) <= np.absolute(results.value))
                diff_sig = perfect_score_adjustment(ds_1.value, ds_2.value, self.statistic, pval)

            results.value = diff_sig

        return results

    def _get_bootstrapped_stats(self, series_data, axis="1"):
        """ Calculates aggregation statistic value and CI intervals if needed for input data
            Args:
                series_data: pandas data frame
            Returns:
                BootstrapDistributionResults object

        """

        # if the data frame is empty - do nothing and return an empty object
        if series_data.empty:
            return BootstrapResults(lower_bound=None,
                                    value=None,
                                    upper_bound=None)
        # check if derived series are present
        has_derived_series = False
        if self.params['derived_series_' + axis]:
            has_derived_series = True

        # sort data by dates
        series_data = sort_data(series_data)
        series_data.reset_index(inplace=True, drop=True)

        if 'line_type' in self.params.keys() and self.params['line_type'] is not None and self.params['line_type'] != 'None':
            # find the function that prepares data and execute it
            func = getattr(self, f"_prepare_{self.params['line_type']}_data")
            func(series_data)

        # input data has to be in numpy format for bootstrapping
        data = series_data.to_numpy()

        if self.params['num_iterations'] == 1:
            # don't need bootstrapping and CI calculation -
            # calculate the statistic and exit
            stat_val = self._calc_stats(data)[0]
            results = BootstrapResults(lower_bound=None,
                                       value=stat_val,
                                       upper_bound=None)
            # save original data only if we need it in the future
            # for derived series calculation
            if has_derived_series:
                results.set_original_values(data)
        else:
            # need bootstrapping and CI calculation in addition to statistic
            try:
                block_length = 1
                # to use circular block bootstrap or not
                is_cbb = True
                if 'circular_block_bootstrap' in self.params.keys():
                    is_cbb = parse_bool(self.params['circular_block_bootstrap'])

                if is_cbb:
                    block_length = int(math.sqrt(len(data)))
                results = bootstrap_and_value(
                    data,
                    stat_func=self._calc_stats,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    save_data=has_derived_series,
                    block_length=block_length)

            except KeyError as err:
                results = BootstrapResults(None, None, None)
                print(err)
        return results

    def _validate_series_cases_for_derived_operation(self, series_data, axis="1", num_diff_vals=1):
        """ Checks if the derived curve can be calculated.
            The criteria - input array must have only unique
            (fcst_valid, fcst_lead, stat_name) cases.
            Can't calculate differences if  multiple values for one valid date/fcst_lead


            Args:
                series_data: 2d numpu array
                axis: axis of the series
                num_diff_vals: number of values in the group if the series has a group,
                    1 - otherwise
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
            unique_date_size = len(set(map(tuple, date_lead_stat)))
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

        if len(series_data) / num_diff_vals != unique_date_size \
                and self.params['list_stat_' + axis] not in self.EXEMPTED_VARS:
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
        result = pd.DataFrame()
        row_number = len(series)
        # fill series variables and values
        for field_ind, field in enumerate(series_fields):
            result[field] = [row[field_ind] for row in series]

        # fill the stats  and CI values placeholders with None
        result['fcst_var'] = [None] * row_number
        result['stat_value'] = [None] * row_number
        result['stat_btcl'] = [None] * row_number
        result['stat_btcu'] = [None] * row_number
        result['nstats'] = [None] * row_number
        return result

    def _get_derived_points(self, series_val, indy_vals, axis="1"):
        """identifies and returns as an list all possible derived points values

            Args:
                series_val: dictionary of series variable to values
                indy_vals: list of independent values
            Returns: a list of all possible values for the each derived points

        """

        # for each derived series
        result = []
        for derived_serie in self.params['derived_series_' + axis]:
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

    def _proceed_with_axis(self, axis="1"):
        """Calculates stat values for the requested Y axis

            Args:
                axis: 1 or 2 Y axis
             Returns:
                pandas dataframe  with calculated stat values and CI

        """
        if not self.input_data.empty:
            # replace thresh_i values for reliability plot
            indy_vals = self.params['indy_vals']
            if self.params['indy_var'] == 'thresh_i' and self.params['line_type'] == 'pct':
                indy_vals_int = self.input_data['thresh_i'].tolist()
                indy_vals_int.sort()
                indy_vals_int = np.unique(indy_vals_int).tolist()
                indy_vals = list(map(str, indy_vals_int))

            # identify all possible points values by adding series values, indy values
            # and statistics and then permute them
            series_val = self.params['series_val_' + axis]
            all_fields_values = series_val.copy()
            if indy_vals:
                all_fields_values[self.params['indy_var']] = indy_vals
            all_fields_values['stat_name'] = self.params['list_stat_' + axis]
            all_points = list(itertools.product(*all_fields_values.values()))

            if self.params['derived_series_' + axis]:
                # identifies and add all possible derived points values
                all_points.extend(self._get_derived_points(series_val, indy_vals, axis))

            # init the template for output frame
            out_frame = self._init_out_frame(all_fields_values.keys(), all_points)

            point_to_distrib = {}

            # for each point
            for point_ind, point in enumerate(all_points):
                # get statistic. Use reversed because it is more likely that the stat is in the end
                for component in reversed(point):
                    if component in set(self.params['list_stat_' + axis]):
                        self.statistic = component.lower()
                        break
                is_derived = is_derived_point(point)
                if not is_derived:
                    # filter point data
                    all_filters = []
                    all_filters_pct = []
                    filters_wihtout_indy = []
                    indy_val = None
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
                            if field != self.params['indy_var']:  #
                                filters_wihtout_indy. \
                                    append((self.input_data[field].isin(filter_list)))
                            else:
                                indy_val = filter_value

                            all_filters.append(self.input_data[field].isin(filter_list))
                        if field in series_val.keys():
                            all_filters_pct.append((self.input_data[field].isin(filter_list)))

                    # add fcst var
                    fcst_var = None
                    if len(self.params['fcst_var_val_' + axis]) > 0:
                        fcst_var = list(self.params['fcst_var_val_' + axis].keys())[0]
                        if 'fcst_var' in self.input_data.columns:
                            all_filters.append((self.input_data['fcst_var'].isin([fcst_var])))

                    # use numpy to select the rows where any record evaluates to True
                    mask = np.array(all_filters).all(axis=0)
                    point_data = self.input_data.loc[mask]

                    if self.params['line_type'] == 'pct':
                        if all_filters_pct:
                            mask_pct = np.array(all_filters_pct).all(axis=0)
                            point_data_pct = self.input_data.loc[mask_pct]
                        else:
                            point_data_pct = self.input_data
                        # collect all columns that starts with oy_i and on_i
                        filter_oy_i = [col for col in point_data_pct if col.startswith('oy_i')]
                        filter_on_i = [col for col in point_data_pct if col.startswith('on_i')]
                        # calculate oy_total
                        oy_total = point_data_pct[filter_oy_i].values.sum()

                        # calculate T
                        sum_n_i_orig_T = point_data_pct[filter_on_i].values.sum() + oy_total

                        # calculate o_bar
                        o_bar = oy_total / sum_n_i_orig_T

                        point_data.insert(len(point_data.columns), 'T', sum_n_i_orig_T)
                        point_data.insert(len(point_data.columns), 'oy_total', oy_total)
                        point_data.insert(len(point_data.columns), 'o_bar', o_bar)

                    # aggregate point data
                    series_var_val = self.params['series_val_' + axis]
                    if any(';' in series_val for series_val in series_var_val):
                        point_data = aggregate_field_values(series_var_val, point_data, self.params['line_type'])
                    elif indy_val and ';' in indy_val:
                        # if aggregated value in indy val - add it to series values add aggregate
                        series_indy_var_val = series_var_val
                        series_indy_var_val[self.params['indy_var']] = [indy_val]
                        point_data = aggregate_field_values(series_indy_var_val, point_data,
                                                            self.params['line_type'])
                    # calculate bootstrap results
                    bootstrap_results = self._get_bootstrapped_stats(point_data, axis)
                    # save bootstrap results
                    point_to_distrib[point] = bootstrap_results
                    n_stats = len(point_data)

                else:
                    # calculate bootstrap results for the derived point
                    bootstrap_results = self._get_bootstrapped_stats_for_derived(
                        point,
                        point_to_distrib,
                        axis)
                    n_stats = 0

                # save results to the output data frame
                out_frame['fcst_var'][point_ind] = fcst_var
                out_frame['stat_value'][point_ind] = bootstrap_results.value
                out_frame['stat_btcl'][point_ind] = bootstrap_results.lower_bound
                out_frame['stat_btcu'][point_ind] = bootstrap_results.upper_bound
                out_frame['nstats'][point_ind] = n_stats

        else:
            out_frame = pd.DataFrame()
        return out_frame

    def calculate_stats_and_ci(self):
        """ Calculates aggregated statistics and confidants intervals
            ( if parameter num_iterations > 1) for each series point
            Writes output data to the file

        """

        # set random seed if present
        if self.params['random_seed'] is not None and self.params['random_seed'] != 'None':
            np.random.seed(self.params['random_seed'])

        is_event_equal = parse_bool(self.params['event_equal'])

        if self.params['line_type'] == 'pct':
            self.column_names = np.append(self.column_names, 'T')
            self.column_names = np.append(self.column_names, 'oy_total')
            self.column_names = np.append(self.column_names, 'o_bar')

        # perform grouping
        series_val = self.params['series_val_1']
        group_to_value_index = 1
        if series_val:
            for key in series_val.keys():
                for val in series_val[key]:
                    if GROUP_SEPARATOR in val:
                        new_name = 'Group_y1_' + str(group_to_value_index)
                        self.group_to_value[new_name] = val
                        group_to_value_index = group_to_value_index + 1

        series_val = self.params['series_val_2']
        if series_val:
            group_to_value_index = 1
            if series_val:
                for key in series_val.keys():
                    for val in series_val[key]:
                        if GROUP_SEPARATOR in val:
                            new_name = 'Group_y2_' + str(group_to_value_index)
                            self.group_to_value[new_name] = val
                            group_to_value_index = group_to_value_index + 1

        # perform EE if needed
        if is_event_equal:
            self.input_data = perform_event_equalization(self.params, self.input_data)

        # get results for axis1
        out_frame = self._proceed_with_axis("1")

        # get results for axis2 if needed
        if self.params['series_val_2']:
            out_frame = out_frame.append(self._proceed_with_axis("2"))

        header = True
        mode = 'w'

        if 'append_to_file' in self.params.keys() and self.params['append_to_file'] == 'True':
            header = False
            mode = 'a'

        export_csv = out_frame.to_csv(self.params['agg_stat_output'],
                                      index=None, header=header, mode=mode,
                                      sep="\t", na_rep="NA", float_format='%.'+ str(PRECISION) +'f')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    PARAMS = yaml.load(ARGS.parameters_file, Loader=yaml.FullLoader)

    AGG_STAT = AggStat(PARAMS)
    AGG_STAT.calculate_stats_and_ci()
