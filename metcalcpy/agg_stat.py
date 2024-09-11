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
from metcalcpy.util.utils import  get_met_version

from metcalcpy.util.utils import is_string_integer, get_derived_curve_name, \
    calc_derived_curve_value, intersection, is_derived_point, parse_bool, \
    OPERATION_TO_SIGN, perfect_score_adjustment, perform_event_equalization, \
    aggregate_field_values, sort_data, DerivedCurveComponent, is_string_strictly_float

from metcalcpy.logging_config import setup_logging

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
        self.logger = setup_logging(in_params)
        logger = self.logger
        logger.debug("Initializing AggStat with parameters")
        self.statistic = None
        self.derived_name_to_values = {}
        self.params = in_params

        try:
            self.input_data = pd.read_csv(
                self.params['agg_stat_input'],
                header=[0],
                sep='\t'
            )
            logger.info(f"Successfully loaded data from {self.params['agg_stat_input']}")
            cols = self.input_data.columns.to_list()
            # Convert all col headers to lower case
            lc_cols = [lc_cols.lower() for lc_cols in cols]
            self.column_names = np.array(lc_cols)
            self.input_data.columns = lc_cols
        except pd.errors.EmptyDataError as e:
            logger.error("Input data file is empty, raising EmptyDataError.", exc_info=True)
            raise
        except KeyError as e:
            logger.error(f"Parameter with key {str(e)} is missing, raising KeyError.", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error occurred during data loading: {str(e)}", exc_info=True)
            raise
        self.group_to_value = {}
        logger.debug("AggStat initialized successfully.")

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
        'vcnt_dir_me': ['dir_me'],
        'vcnt_dir_mae': ['dir_mae'],
        'vcnt_dir_mse': ['dir_mse'],
        'vcnt_dir_rmse': ['dir_mse'],

        'vl1l2_bias': ['uvffbar', 'uvoobar'],
        'vl1l2_fvar': ['uvffbar', 'f_speed_bar'],
        'vl1l2_ovar': ['uvoobar', 'o_speed_bar'],
        'vl1l2_speed_err': ['ufbar', 'vfbar', 'uobar', 'vobar'],
        'vl1l2_rmsve': ['uvffbar', 'uvfobar', 'uvoobar'],
        'vl1l2_msve': ['uvffbar', 'uvfobar', 'uvoobar'],
        'vl1l2_dir_me': ['dir_me'],
        'vl1l2_dir_mae': ['dir_mae'],
        'vl1l2_dir_mse': ['dir_mse'],


        'val1l2_anom_corr':
            ['ufabar', 'vfabar', 'uoabar', 'voabar', 'uvfoabar', 'uvffabar', 'uvooabar'],
        'val1l2_dira_me': ['dira_me'],
        'val1l2_dira_mae': ['dira_mae'],
        'val1l2_dira_mse': ['dira_mse'],

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
        'ecnt_bias_ratio': ['me_ge_obs', 'me_lt_obs'],
        'ecnt_ign_conv_oerr': ['ign_conv_oerr'],
        'ecnt_ign_corr_oerr': ['ign_corr_oerr'],

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
        logger = self.logger
        func_name = f'calculate_{self.statistic}'
        try:
            stat_function = globals()[func_name]
        except KeyError:
            logger.error(f"Statistical function {func_name} not found in globals.")
            raise KeyError(f"Function {func_name} not defined.")
        # some functions have an extra 3rd parameter that represents
        # if some data preliminary data aggregation was done
        # if this parameter is present we need to add it
     
        num_parameters = len(signature(globals()[func_name]).parameters)
        logger.debug(f"Function {func_name} expects {num_parameters} parameters.")
    
        if values is None:
            logger.error("Input values array is None.")
            raise ValueError("Input values cannot be None.")

        if values is not None and values.ndim == 2:
            logger.debug("Processing single value case for statistical calculation.")
            # The single value case
            try:
                if num_parameters == 2:
                    stat_values = [stat_function(values, self.column_names)]
                else:
                    stat_values = [stat_function(values, self.column_names, True)]
                logger.info("Statistics calculated successfully for single value case.")
            except Exception as e:
                logger.error(f"Failed to calculate statistics: {e}", exc_info=True)
                raise

        elif values is not None and values.ndim == 3:
            # bootstrapped case
            logger.debug("Processing bootstrapped case for statistical calculation.")
            stat_values = []
            try:
                for row in values:
                    if num_parameters == 2:
                        stat_value = [stat_function(row, self.column_names)]
                    else:
                        stat_value = [stat_function(row, self.column_names, True)]
                    stat_values.append(stat_value)
                logger.info("Statistics calculated successfully for all bootstrap samples.")
            except Exception as e:
                logger.error(f"Failed during bootstrap calculations: {e}", exc_info=True)
                raise

            # pool = mp.Pool(mp.cpu_count())
            # stat_values = pool.map(partial(globals()['calculate_{}'.format(stat)],
            # columns_names=columns_names), [row for row in data_for_stats])
            # pool.close()
            # pool.join()

        else:
            logger.error(f"Invalid data dimensions {values.ndim}; expected 2D or 3D array.")
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
        logger = self.logger
        logger.debug("Starting calculation of derived statistics.")
        
        if values_both_arrays is None:
            logger.error("Input values array is None.")
            raise ValueError("Input values cannot be None.")

        if values_both_arrays.ndim not in [2, 3]:
            logger.error(f"Invalid data dimensions {values_both_arrays.ndim}; expected 2D or 3D array.")
            raise KeyError("Invalid data dimensions")

        if values_both_arrays is not None and values_both_arrays.ndim == 2:
            # The single value case
            num_of_columns = values_both_arrays.shape[1] - 1
            # get values for the 1st array
            values_1 = values_both_arrays[:, 0:int(num_of_columns / 2)]
            # get values for the 2nd array
            values_2 = values_both_arrays[:, int(num_of_columns / 2):num_of_columns]

            try:
                # find the index of the stat column
                stat_column_index = np.where(self.column_names == 'stat_name')[0][0]
                # find the actual statistic and corresponding functions for both curves
                stat_1 = values_1[0, stat_column_index].lower()
                stat_2 = values_2[0, stat_column_index].lower()
                func_name_1 = f'calculate_{stat_1}'
                func_name_2 = f'calculate_{stat_2}'
            except ValueError:
                func_name_1 = f'calculate_{self.statistic}'
                func_name_2 = f'calculate_{self.statistic}'
                logger.error(f"Error finding statistics function: {e}")
                raise ValueError("Error processing statistic names")             
 

            # some functions have an extra 3rd parameter that represents
            # if some data preliminary data aggregation was done
            # if this parameter is present we need to add it
            num_parameters_1 = len(signature(globals()[func_name_1]).parameters)
            num_parameters_2 = len(signature(globals()[func_name_2]).parameters)


            # calculate stat for the 1st array
            try:
                if num_parameters_1 == 2:
                    stat_values_1 = [globals()[func_name_1](values_1, self.column_names)]
                else:
                    stat_values_1 = [globals()[func_name_1](values_1, self.column_names, True)]

                # calculate stat for the 2nd array
                if num_parameters_2 == 2:
                    stat_values_2 = [globals()[func_name_2](values_2, self.column_names)]
                else:
                    stat_values_2 = [globals()[func_name_2](values_2, self.column_names, True)]
            except Exception as e:
                logger.error(f"Error calculating statistics: {e}")
                raise

            # calculate derived stat
            try:
                stat_values = calc_derived_curve_value(
                    stat_values_1,
                    stat_values_2,
                    values_both_arrays[0, -1])
                if not isinstance(stat_values, list):
                    stat_values = [stat_values]
            except Exception as e:
                logger.error(f"Error calculating derived statistics: {e}", exc_info=True)
            raise

        elif values_both_arrays is not None and values_both_arrays.ndim == 3:
            # bootstrapped case
            stat_values = []
            num_of_columns = values_both_arrays.shape[2] - 1
            for row in values_both_arrays:

                # get values for the 1st array
                values_1 = row[:, 0:int(num_of_columns / 2)]
                # get values for the 2nd array
                values_2 = row[:, int(num_of_columns / 2):num_of_columns]

                try:
                    # find the index of the stat column
                    stat_column_index = np.where(self.column_names == 'stat_name')[0][0]
                    # find the actual statistic and corresponding functions for both curves
                    stat_1 = values_1[0, stat_column_index].lower()
                    stat_2 = values_2[0, stat_column_index].lower()
                    func_name_1 = f'calculate_{stat_1}'
                    func_name_2 = f'calculate_{stat_2}'
                except ValueError:
                    func_name_1 = f'calculate_{self.statistic}'
                    func_name_2 = f'calculate_{self.statistic}'
                    logger.error(f"Error finding statistics function: {e}")
                    raise ValueError("Error processing statistic names")

                # some functions have an extra 3rd parameter that represents
                # if some data preliminary data aggregation was done
                # if this parameter is present we need to add it
                num_parameters_1 = len(signature(globals()[func_name_1]).parameters)
                num_parameters_2 = len(signature(globals()[func_name_2]).parameters)

                # calculate stat for the 1st array
                try:
                    if num_parameters_1 == 2:
                        stat_values_1 = [globals()[func_name_1](values_1, self.column_names)]
                    else:
                        stat_values_1 = [globals()[func_name_1](values_1, self.column_names, True)]

                    # calculate stat for the 2nd array
                    if num_parameters_2 == 2:
                        stat_values_2 = [globals()[func_name_2](values_2, self.column_names)]
                    else:
                        stat_values_2 = [globals()[func_name_2](values_2, self.column_names, True)]

                except Exception as e:
                    logger.error(f"Error calculating statistics: {e}")
                    raise

                # calculate derived stat
                try:
                    stat_value = calc_derived_curve_value(
                        stat_values_1,
                        stat_values_2,
                        row[0, -1])
                    if not isinstance(stat_value, list):
                        stat_value = [stat_value]
                    stat_values.append(stat_value)
                    logger.info("Derived statistics calculated successfully.")
                except Exception as e:
                    logger.error(f"Error calculating derived statistics: {e}", exc_info=True)
                    raise

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
        logger = self.logger
        logger.debug("Starting preparation of sl1l2 data.")
       
        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            try:
                for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                    if column in data_for_prepare.columns and 'total' in data_for_prepare.columns:
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total'].values
                        logger.debug(f"Data for column '{column}' multiplied by 'total'.")
                    else:
                        logger.warning(f"Column '{column}' or 'total' not found in the DataFrame.")
            except Exception as e:
                logger.error(f"Failed to prepare data for statistic calculation: {e}", exc_info=True)
                raise
        else:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        logger.info("sl1l2 data preparation completed successfully.")

    def _prepare_sal1l2_data(self, data_for_prepare):
        """Prepares sal1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug(f"Starting preparation of sal1l2 data for statistic '{self.statistic}'.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        if self.statistic not in self.STATISTIC_TO_FIELDS:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        try:
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                if column in data_for_prepare.columns and 'total' in data_for_prepare.columns:
                    data_for_prepare[column] = data_for_prepare[column] * data_for_prepare['total']
                    logger.debug(f"Column '{column}' successfully multiplied by 'total'.")
                else:
                    missing_columns = [col for col in [column, 'total'] if col not in data_for_prepare.columns]
                    logger.warning(f"Missing columns {missing_columns} in DataFrame. Multiplication skipped.")
        except Exception as e:
            logger.error(f"Failed to prepare data for statistic calculation: {e}", exc_info=True)
            raise

        logger.info("sal1l2 data preparation completed successfully.")
    
    def _prepare_grad_data(self, data_for_prepare):
        """Prepares grad data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug(f"Starting preparation of grad data for statistic '{self.statistic}'.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        if self.statistic not in self.STATISTIC_TO_FIELDS:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        try:
            for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                if column in data_for_prepare.columns and 'total' in data_for_prepare.columns:
                    data_for_prepare[column] = data_for_prepare[column] * data_for_prepare['total']
                    logger.debug(f"Column '{column}' successfully multiplied by 'total'.")
                else:
                    missing_columns = [col for col in [column, 'total'] if col not in data_for_prepare.columns]
                    logger.warning(f"Missing columns {missing_columns} in DataFrame. Multiplication skipped.")
        except Exception as e:
            logger.error(f"Failed to prepare data for statistic calculation: {e}", exc_info=True)
            raise

        logger.info("Grad data preparation completed successfully.")

    def _prepare_vl1l2_data(self, data_for_prepare):
        """Prepares vl1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total' value
            or 'total_dir' value for MET version 12.0 and above.

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of vl1l2 data.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        # Determine the MET version for this data.  If MET v12.0 or above, use the 'total_dir' column rather than
        # the 'total' column.
        try:
            met_version = get_met_version(data_for_prepare)
            major = int(met_version.major)
            logger.debug(f"Detected MET version: {major}")
        except Exception as e:
            logger.error(f"Failed to determine MET version from data: {e}", exc_info=True)
            raise

        if self.statistic not in self.STATISTIC_TO_FIELDS:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            try:
                for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                    if major >= int(12):
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total_dir'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total_dir'.")
                    else:
                        data_for_prepare[column] \
                        = data_for_prepare[column].values * data_for_prepare['total'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total'.")
            except Exception as e:
                logger.error(f"Error during data preparation: {e}", exc_info=True)
                raise

        logger.info("vl1l2 data preparation completed successfully.")

    def _prepare_val1l2_data(self, data_for_prepare):
        """Prepares val1l2 data.
            Multiplies needed for the statistic calculation columns to the 'total_dir' value
            (MET 12.0) or 'total' MET<12.0

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        # Determine the MET version for this data.  If MET v12.0 or above, use the 'total_dir' column rather than
        # the 'total' column.
        logger = self.logger
        logger.debug("Starting preparation of val1l2 data.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        try:
            met_version = get_met_version(data_for_prepare)
            major = int(met_version.major)
            logger.debug(f"Detected MET version: {major}")
        except Exception as e:
            logger.error(f"Failed to determine MET version from data: {e}", exc_info=True)
            raise

        if self.statistic not in self.STATISTIC_TO_FIELDS:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            try:
                for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                    if major >= int(12):
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total_dir'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total_dir'.")
                    else:
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total'.")
            except Exception as e:
                logger.error(f"Error during data preparation: {e}", exc_info=True)
                raise

        logger.info("val1l2 data preparation completed successfully.")

    def _prepare_vcnt_data(self, data_for_prepare):
        """Prepares vcnt data.
            Multiplies needed for the statistic calculation columns to the 'total_dir' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of vcnt data.")
        # Determine the MET version for this data.  If MET v12.0 or above, use the 'total_dir' column rather than
        # the 'total' column.
        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        try:
            met_version = get_met_version(data_for_prepare)
            major = int(met_version.major)
            logger.debug(f"Detected MET version: {major}")
        except Exception as e:
            logger.error(f"Failed to determine MET version from data: {e}", exc_info=True)
            raise

        if self.statistic not in self.STATISTIC_TO_FIELDS:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        if self.statistic in self.STATISTIC_TO_FIELDS.keys():
            try:
                for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                    if major >= int(12):
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total_dir'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total_dir'.")
                    else:
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total'.")
            except Exception as e:
                logger.error(f"Error during data preparation: {e}", exc_info=True)
                raise

        logger.info("vcnt data preparation completed successfully.")

    def _prepare_ecnt_data(self, data_for_prepare):
        """Prepares ecnt data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of ECNT data.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        try:
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
            logger.debug("Basic statistical calculations completed.")

            if self.statistic in self.STATISTIC_TO_FIELDS.keys():
                for column in self.STATISTIC_TO_FIELDS[self.statistic]:
                    if column == 'me_ge_obs':
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['n_ge_obs'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'n_ge_obs'.")
                    elif column == 'me_lt_obs':
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['n_lt_obs'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'n_lt_obs'.")
                    else:
                        data_for_prepare[column] \
                            = data_for_prepare[column].values * data_for_prepare['total'].values
                        logger.debug(f"Column '{column}' successfully multiplied by 'total'.")
            else:
                logger.warning(f"Statistic '{self.statistic}' does not have associated fields for ECNT preparation.")
            logger.info("ECNT data preparation completed successfully.")

        except KeyError as e:
            logger.error(f"Key error during data preparation: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data preparation: {e}", exc_info=True)
            raise

    def _prepare_rps_data(self, data_for_prepare):
        """Prepares rps data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of RPS data.")

        
        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        try:
            total = data_for_prepare['total'].values
            d_rps_climo = data_for_prepare['rps'].values / (1 - data_for_prepare['rpss'].values)
            data_for_prepare['rps_climo'] = d_rps_climo * total
            logger.debug(f"Column 'rps_climo' successfully calculated.")
            data_for_prepare['rps'] = data_for_prepare['rps'].values * total
            logger.debug(f"Column 'rps' successfully multiplied by 'total'.")
            data_for_prepare['rps_comp'] = data_for_prepare['rps_comp'].values * total
            logger.debug(f"Column 'rps_comp' successfully multiplied by 'total'.")
            self.column_names = data_for_prepare.columns.values
        except KeyError as e:
            logger.error(f"Key error during data preparation: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data preparation: {e}", exc_info=True)
            raise

    def _prepare_ssvar_data(self, data_for_prepare):
        """Prepares ssvar data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug(f"Starting preparation of ssvar data for statistic '{self.statistic}'.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        # rename bin_n column to total
        data_for_prepare.rename(columns={"total": "total_orig", "bin_n": "total"}, inplace=True)
        self.column_names = data_for_prepare.columns.values

        if self.statistic not in self.STATISTIC_TO_FIELDS:
            error_message = f"Statistic '{self.statistic}' is not recognized or lacks associated fields."
            logger.error(error_message)
            raise KeyError(error_message)

        for column in self.STATISTIC_TO_FIELDS[self.statistic]:
            data_for_prepare[column] \
                = data_for_prepare[column].values * data_for_prepare['total'].values
            logger.debug(f"Column '{column}' successfully multiplied by 'total'.")

        logger.info("ssvar data preparation completed successfully.")

    def _prepare_nbr_cnt_data(self, data_for_prepare):
        """Prepares nbrcnt data.
            Multiplies needed for the statistic calculation columns to the 'total' value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug(f"Starting preparation of nbrcnt data for statistic '{self.statistic}'.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        total = data_for_prepare['total'].values
        fbs = total * data_for_prepare['fbs'].values
        fss_den = (data_for_prepare['fbs'].values / (1.0 - data_for_prepare['fss'].values)) * total

        f_rate = total * data_for_prepare['f_rate'].values
        data_for_prepare['fbs'] = fbs
        data_for_prepare['fss'] = fss_den
        data_for_prepare['f_rate'] = f_rate

        logger.info("nbrcnt data preparation completed successfully.")

    def _prepare_pct_data(self, data_for_prepare):
        """Prepares pct data.
            Multiplies needed for the statistic calculation columns to the 'total'value

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """


    def _prepare_mctc_data(self, data_for_prepare):
        """Prepares mctc data.
           Nothingneeds to be done

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of MCTC data.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        if 'ec_value' in data_for_prepare.columns:
            if not (data_for_prepare['ec_value'] == data_for_prepare['ec_value'][0]).all():
                raise ValueError('EC_VALUE is NOT constant across  MCTC lines')


    def _prepare_ctc_data(self, data_for_prepare):
        """Prepares CTC data.
            Checks if all values from ec_value column are the same and if not - throws an error

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of CTC data.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

        if 'ec_value' in data_for_prepare.columns:
            if not (data_for_prepare['ec_value'] == data_for_prepare['ec_value'][0]).all():
                raise ValueError('EC_VALUE is NOT constant across  CTC lines')


    def _prepare_cts_data(self, data_for_prepare):
        """Prepares cts data.
            Checks if all values from ec_value column are the same and if not - throws an error

            Args:
                data_for_prepare: a 2d numpy array of values we want to calculate the statistic on
        """
        logger = self.logger
        logger.debug("Starting preparation of CTS data.")

        if data_for_prepare is None:
            logger.error("Input data for preparation is None.")
            raise ValueError("Input data cannot be None.")

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
        """
        Calculates aggregation derived statistic value and CI intervals if needed for input data.
        
        Args:
            series: array of length = 3 where
                1st element - derived series title,
                others  - additional values like indy val and statistic.
            distributions: dictionary of the series title to its BootstrapDistributionResult object.
            
        Returns:
            BootstrapDistributionResults object.
        """
        logger = self.logger
        logger.debug("Starting bootstrapped statistics calculation for derived series.")

        # get derived name
        derived_name = ''
        for operation in OPERATION_TO_SIGN:
            for point_component in series:
                if point_component.startswith((operation + '(', operation + ' (')):
                    derived_name = point_component
                    break
        logger.debug(f"Derived name identified: {derived_name}")

        try:
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

        except KeyError as err:
            logger.error(f"Error during derived component lookup: {err}", exc_info=True)
            return BootstrapResults(None, None, None)
            
        logger.debug(f"First series components: {permute_for_first_series}")
        logger.debug(f"Second series components: {permute_for_second_series}")

        ds_1 = ds_2 = None

        # for each component find its BootstrapDistributionResult object
        for series_to_distrib_key in distributions.keys():
            if all(elem in permute_for_first_series for elem in series_to_distrib_key):
                ds_1 = distributions[series_to_distrib_key]
            if all(elem in permute_for_second_series for elem in series_to_distrib_key):
                ds_2 = distributions[series_to_distrib_key]
            if ds_1 is not None and ds_2 is not None:
                break

        if ds_1 is None or ds_2 is None:
            logger.warning("Could not find BootstrapDistributionResult objects for one or both series.")
        
        # if BootstrapDistributionResult object doesn't exist or the original series data size is 0, return empty object
        if ds_1.values is None or ds_2.values is None or ds_1.values.size == 0 or ds_2.values.size == 0:
            logger.warning("One or both series have no values. Returning empty BootstrapResults object.")
            return BootstrapResults(lower_bound=None, value=None, upper_bound=None)

        # calculate the number of values in the group if the series has a group, needed for validation
        num_diff_vals_first = sum(len(val.split(GROUP_SEPARATOR)) for val in permute_for_first_series if len(val.split(GROUP_SEPARATOR)) > 1)
        num_diff_vals_second = sum(len(val.split(GROUP_SEPARATOR)) for val in permute_for_second_series if len(val.split(GROUP_SEPARATOR)) > 1)
        num_diff_vals_first = max(num_diff_vals_first, 1)
        num_diff_vals_second = max(num_diff_vals_second, 1)

        # validate data
        if derived_curve_component.derived_operation != 'SINGLE':
            logger.debug("Validating series for derived operation.")
            self._validate_series_cases_for_derived_operation(ds_1.values, axis, num_diff_vals_first)
            self._validate_series_cases_for_derived_operation(ds_2.values, axis, num_diff_vals_second)

        # handle bootstrapping
        if self.params['num_iterations'] == 1 or derived_curve_component.derived_operation == 'ETB':
            logger.debug("No bootstrapping required; calculating derived statistic.")
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

            stat_val = calc_derived_curve_value(ds_1_value, ds_2_value, derived_curve_component.derived_operation)
            if stat_val is not None:
                results = BootstrapResults(lower_bound=None, value=round_half_up(stat_val[0], 5), upper_bound=None)
            else:
                results = BootstrapResults(lower_bound=None, value=None, upper_bound=None)
            results.set_distributions([results.value])
        else:
            logger.debug("Performing bootstrapping with CI calculation.")
            operation = np.full((len(ds_1.values), 1), derived_curve_component.derived_operation)
            values_both_arrays = np.concatenate((ds_1.values, ds_2.values, operation), axis=1)

            try:
                block_length = int(math.sqrt(len(values_both_arrays))) if 'circular_block_bootstrap' in self.params and parse_bool(self.params['circular_block_bootstrap']) else 1
                results = bootstrap_and_value(
                    values_both_arrays,
                    stat_func=self._calc_stats_derived,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    alpha=self.params['alpha'],
                    save_data=False,
                    save_distributions=(derived_curve_component.derived_operation == 'DIFF_SIG'),
                    block_length=block_length)
            except KeyError as err:
                logger.error(f"Error during bootstrapping: {err}", exc_info=True)
                return BootstrapResults(None, None, None)

        # Post-processing for DIFF_SIG
        if derived_curve_component.derived_operation == 'DIFF_SIG':
            logger.debug("Processing DIFF_SIG operation for derived statistics.")
            distributions = [i for i in results.distributions if i is not None]
            if distributions and results.value is not None:
                distribution_mean = np.mean(distributions)
                distribution_under_h0 = distributions - distribution_mean
                pval = np.mean(np.absolute(distribution_under_h0) <= np.absolute(results.value))
                diff_sig = perfect_score_adjustment(ds_1.value, ds_2.value, self.statistic, pval)
                results.value = diff_sig

        logger.info("Completed derived statistics calculation.")
        return results

    def _get_bootstrapped_stats(self, series_data, axis="1"):
        """ Calculates aggregation statistic value and CI intervals if needed for input data
            Args:
                series_data: pandas data frame
            Returns:
                BootstrapDistributionResults object

        """
        logger = self.logger
        logger.debug("Starting bootstrapped statistics calculation.")

        # Check if the data frame is empty
        if series_data.empty:
            logger.warning("Input series data is empty. Returning empty BootstrapResults.")
            return BootstrapResults(lower_bound=None, value=None, upper_bound=None)

        # Check for derived series
        has_derived_series = False
        if self.params.get('derived_series_' + axis):
            has_derived_series = True
            logger.debug("Derived series found for axis '%s'.", axis)

        # Sort data by dates and reset index
        logger.debug("Sorting series data.")
        series_data = sort_data(series_data)
        series_data.reset_index(inplace=True, drop=True)
        logger.debug("Data sorting completed and index reset.")

        # Prepare data for specific line type if present
        if 'line_type' in self.params and self.params['line_type']:
            logger.debug("Preparing data for line type '%s'.", self.params['line_type'])
            func = getattr(self, f"_prepare_{self.params['line_type']}_data")
            func(series_data)
            logger.debug("Data preparation for line type '%s' completed.", self.params['line_type'])

        # Convert data to numpy format for bootstrapping
        logger.debug("Converting series data to numpy format.")
        data = series_data.to_numpy()

        # Perform calculation without bootstrapping if only one iteration
        if self.params['num_iterations'] == 1:
            logger.debug("No bootstrapping needed (num_iterations = 1). Calculating statistics.")
            try:
                stat_val = self._calc_stats(data)[0]
                results = BootstrapResults(lower_bound=None, value=stat_val, upper_bound=None)
                logger.info("Statistic calculated without bootstrapping. Value: %s", stat_val)

                # Save original data for derived series if needed
                if has_derived_series:
                    logger.debug("Saving original data for derived series.")
                    results.set_original_values(data)
            except Exception as e:
                logger.error("Error during statistic calculation: %s", e, exc_info=True)
                raise
        else:
            # Bootstrapping required with CI calculation
            logger.debug("Bootstrapping with %d iterations.", self.params['num_iterations'])
            try:
                block_length = 1
                # Determine whether to use circular block bootstrap
                is_cbb = self.params.get('circular_block_bootstrap', True)
                if is_cbb:
                    block_length = int(math.sqrt(len(data)))
                    logger.debug("Using circular block bootstrap with block length: %d", block_length)

                # Perform bootstrapping and CI calculation
                results = bootstrap_and_value(
                    data,
                    stat_func=self._calc_stats,
                    num_iterations=self.params['num_iterations'],
                    num_threads=self.params['num_threads'],
                    ci_method=self.params['method'],
                    save_data=has_derived_series,
                    block_length=block_length
                )
                logger.info("Bootstrapping and CI calculation completed.")

            except KeyError as err:
                logger.error("KeyError during bootstrapping: %s", err, exc_info=True)
                results = BootstrapResults(None, None, None)
            except Exception as e:
                logger.error("Unexpected error during bootstrapping: %s", e, exc_info=True)
                raise

        logger.debug("Bootstrapped statistics calculation completed.")
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
        logger = self.logger
        logger.debug("Starting validation of series for derived operation.")
        logger.debug(f"Axis: {axis}, num_diff_vals: {num_diff_vals}")

        try:
            # Find indexes of columns of interest
            fcst_lead_index = np.where(self.column_names == 'fcst_lead')[0][0]
            stat_name_index = np.where(self.column_names == 'stat_name')[0][0]
            logger.debug(f"fcst_lead_index: {fcst_lead_index}, stat_name_index: {stat_name_index}")

            if "fcst_valid_beg" in self.column_names:
                fcst_valid_ind = np.where(self.column_names == 'fcst_valid_beg')[0][0]
            elif "fcst_valid" in self.column_names:
                fcst_valid_ind = np.where(self.column_names == 'fcst_valid')[0][0]
            elif "fcst_init_beg" in self.column_names:
                fcst_valid_ind = np.where(self.column_names == 'fcst_init_beg')[0][0]
            else:
                fcst_valid_ind = np.where(self.column_names == 'fcst_init')[0][0]

            logger.debug(f"fcst_valid_ind: {fcst_valid_ind}")

            # Filter columns of interest
            date_lead_stat = series_data[:, [fcst_valid_ind, fcst_lead_index, stat_name_index]]
            # Find the number of unique combinations
            unique_date_size = len(set(map(tuple, date_lead_stat)))
            logger.debug(f"Unique date-lead-stat combinations found: {unique_date_size}")

        except TypeError as err:
            logger.error(f"Error during filtering columns: {err}", exc_info=True)
            unique_date_size = []

        # Identify rows with unique combinations
        try:
            ind = np.lexsort(
                (series_data[:, stat_name_index],
                series_data[:, fcst_lead_index], series_data[:, fcst_valid_ind]))
            series_data = series_data[ind, :]
            logger.debug(f"Series data sorted by valid index, lead, and stat name.")
        except Exception as e:
            logger.error(f"Error during sorting series data: {e}", exc_info=True)
            raise

        # Validate if the number of unique combinations matches the data length
        if len(series_data) / num_diff_vals != unique_date_size and self.params['list_stat_' + axis] not in self.EXEMPTED_VARS:
            logger.error("Validation failed. Derived curve can't be calculated due to multiple values for one valid date/fcst_lead.")
            raise NameError("Derived curve can't be calculated. Multiple values for one valid date/fcst_lead")
        
        logger.info("Series validation for derived operation completed successfully.")

    def _init_out_frame(self, series_fields, series):
        """ Initialises the output frame and add series values to each row
            Args:
                series_fields: list of all possible series fields
                series: list of all series definitions
            Returns:
                pandas data frame
        """
        logger = self.logger
        logger.debug("Initializing output frame.")

        # Create an empty DataFrame
        result = pd.DataFrame()

        # Determine the number of rows to be added based on the length of the series
        row_number = len(series)
        logger.debug(f"Number of rows to initialize: {row_number}")

        try:
            # Fill the series variables and values for each field
            for field_ind, field in enumerate(series_fields):
                logger.debug(f"Filling field '{field}' with values from series.")
                result[field] = [row[field_ind] for row in series]

            # Fill the statistical and CI value placeholders with None
            logger.debug("Filling placeholder columns with None values.")
            result['fcst_var'] = [None] * row_number
            result['stat_value'] = [None] * row_number
            result['stat_btcl'] = [None] * row_number
            result['stat_btcu'] = [None] * row_number
            result['nstats'] = [None] * row_number

            logger.info("Output frame initialization completed successfully.")
        except Exception as e:
            logger.error(f"Error during output frame initialization: {e}", exc_info=True)
            raise

        return result

    def _get_derived_points(self, series_val, indy_vals, axis="1"):
        """identifies and returns as an list all possible derived points values

            Args:
                series_val: dictionary of series variable to values
                indy_vals: list of independent values
            Returns: a list of all possible values for the each derived points

        """
        logger = self.logger
        logger.debug("Starting derived points calculation for axis '%s'.", axis)

        result = []

        # Loop through each derived series for the specified axis
        for derived_serie in self.params['derived_series_' + axis]:
            logger.debug(f"Processing derived series: {derived_serie}")

            # Series 1 components
            ds_1 = derived_serie[0].split(' ')
            logger.debug(f"Series 1 components: {ds_1}")

            # Series 2 components
            ds_2 = derived_serie[1].split(' ')
            logger.debug(f"Series 2 components: {ds_2}")

            # Find the variable of the operation by comparing values in each derived series component
            series_var_vals = ()
            for ind, name in enumerate(ds_1):
                if name != ds_2[ind]:
                    series_var_vals = (name, ds_2[ind])
                    logger.debug(f"Identified differing components at index {ind}: {series_var_vals}")
                    break

            # Default to the last key in series_val if no matching variable is found
            series_var = list(series_val.keys())[-1]
            if len(series_var_vals) > 0:
                for var in series_val.keys():
                    if all(elem in series_val[var] for elem in series_var_vals):
                        series_var = var
                        logger.debug(f"Identified series variable: {series_var}")
                        break

            # Create a copy of series_val and modify it for the derived values
            derived_val = series_val.copy()
            derived_val[series_var] = None

            # Filter values based on intersections with ds_1
            for var in series_val.keys():
                if derived_val[var] is not None and intersection(derived_val[var], ds_1) == intersection(derived_val[var], ds_1):
                    derived_val[var] = intersection(derived_val[var], ds_1)
                    logger.debug(f"Updated '{var}' in derived values: {derived_val[var]}")

            # Generate the derived curve name
            derived_curve_name = get_derived_curve_name(derived_serie)
            derived_val[series_var] = [derived_curve_name]
            logger.debug(f"Derived curve name: {derived_curve_name}")

            # If there are independent values, assign them to the appropriate variable
            if len(indy_vals) > 0:
                derived_val[self.params['indy_var']] = indy_vals
                logger.debug(f"Assigned independent values: {indy_vals}")

            # Store the derived series components in a DerivedCurveComponent
            self.derived_name_to_values[derived_curve_name] = DerivedCurveComponent(ds_1, ds_2, derived_serie[-1])

            # Set the stat_name field
            if ds_1[-1] == ds_2[-1]:
                derived_val['stat_name'] = [ds_1[-1]]
            else:
                derived_val['stat_name'] = [ds_1[-1] + "," + ds_2[-1]]
            logger.debug(f"Set 'stat_name' for derived values: {derived_val['stat_name']}")

            # Create all possible combinations of the derived values
            result.append(list(itertools.product(*derived_val.values())))
            logger.debug(f"Derived values appended to result: {derived_val}")

        # Flatten the result list and return it
        flattened_result = [y for x in result for y in x]
        logger.info("Derived points calculation completed. Total derived points: %d", len(flattened_result))
        
        return flattened_result
        
    def _proceed_with_axis(self, axis="1"):
        """Calculates stat values for the requested Y axis

            Args:
                axis: 1 or 2 Y axis
             Returns:
                pandas dataframe  with calculated stat values and CI

        """
        logger = self.logger
        logger.debug(f"Starting to calculate stats for axis: {axis}")

        if not self.input_data.empty:
            logger.debug("Input data is not empty. Proceeding with calculation.")

            # Handle indy_vals for reliability plot if applicable
            indy_vals = self.params['indy_vals']
            if self.params['indy_var'] == 'thresh_i' and self.params['line_type'] == 'pct':
                logger.debug("Replacing thresh_i values for reliability plot.")
                indy_vals_int = self.input_data['thresh_i'].tolist()
                indy_vals_int.sort()
                indy_vals_int = np.unique(indy_vals_int).tolist()
                indy_vals = list(map(str, indy_vals_int))

            # Identify all possible points by adding series values, indy values, and statistics
            series_val = self.params['series_val_' + axis]
            all_fields_values = series_val.copy()
            if indy_vals:
                all_fields_values[self.params['indy_var']] = indy_vals
            all_fields_values['stat_name'] = self.params['list_stat_' + axis]
            all_points = list(itertools.product(*all_fields_values.values()))
            logger.debug(f"Total points identified: {len(all_points)}")

            # Add derived points if present
            if self.params['derived_series_' + axis]:
                logger.debug("Identifying and adding derived points.")
                derived_points = self._get_derived_points(series_val, indy_vals, axis)
                all_points.extend(derived_points)
                logger.debug(f"Total derived points added: {len(derived_points)}")

            # Initialize the output frame
            out_frame = self._init_out_frame(all_fields_values.keys(), all_points)
            logger.debug("Initialized output frame.")

            point_to_distrib = {}

            # Process each point
            for point_ind, point in enumerate(all_points):
                logger.debug(f"Processing point {point_ind + 1}/{len(all_points)}: {point}")
                
                # Determine the statistic for the point
                for component in reversed(point):
                    if component in set(self.params['list_stat_' + axis]):
                        self.statistic = component.lower()
                        break
                logger.debug(f"Statistic identified: {self.statistic}")

                is_derived = is_derived_point(point)

                if not is_derived:
                    logger.debug(f"Processing regular point: {point}")
                    
                    # Filtering point data
                    all_filters = []
                    all_filters_pct = []
                    filters_without_indy = []
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

                        # Convert values to appropriate types
                        for i, filter_val in enumerate(filter_list):
                            if is_string_integer(filter_val):
                                filter_list[i] = int(filter_val)
                            elif is_string_strictly_float(filter_val):
                                filter_list[i] = float(filter_val)

                        # Apply filters
                        if field in self.input_data.keys():
                            if field != self.params['indy_var']:
                                filters_without_indy.append((self.input_data[field].isin(filter_list)))
                            else:
                                indy_val = filter_value
                            all_filters.append(self.input_data[field].isin(filter_list))
                        
                        if field in series_val.keys():
                            all_filters_pct.append((self.input_data[field].isin(filter_list)))

                    # Apply forecast variable filters
                    fcst_var = None
                    if len(self.params['fcst_var_val_' + axis]) > 0:
                        fcst_var = list(self.params['fcst_var_val_' + axis].keys())[0]
                        if 'fcst_var' in self.input_data.columns:
                            all_filters.append((self.input_data['fcst_var'].isin([fcst_var])))

                    # Use numpy to apply filters and select rows
                    mask = np.array(all_filters).all(axis=0)
                    point_data = self.input_data.loc[mask]

                    # Handle percentage line types
                    if self.params['line_type'] == 'pct':
                        logger.debug("Processing percentage line type.")
                        if all_filters_pct:
                            mask_pct = np.array(all_filters_pct).all(axis=0)
                            point_data_pct = self.input_data.loc[mask_pct]
                        else:
                            point_data_pct = self.input_data

                        # Calculate additional metrics for percentage line type
                        filter_oy_i = [col for col in point_data_pct if col.startswith('oy_i')]
                        filter_on_i = [col for col in point_data_pct if col.startswith('on_i')]
                        oy_total = point_data_pct[filter_oy_i].values.sum()
                        sum_n_i_orig_T = point_data_pct[filter_on_i].values.sum() + oy_total
                        o_bar = oy_total / sum_n_i_orig_T
                        point_data.insert(len(point_data.columns), 'T', sum_n_i_orig_T)
                        point_data.insert(len(point_data.columns), 'oy_total', oy_total)
                        point_data.insert(len(point_data.columns), 'o_bar', o_bar)

                    # Aggregate point data if necessary
                    series_var_val = self.params['series_val_' + axis]
                    if any(';' in series_val for series_val in series_var_val):
                        point_data = aggregate_field_values(series_var_val, point_data, self.params['line_type'])
                    elif indy_val and ';' in indy_val:
                        series_indy_var_val = series_var_val
                        series_indy_var_val[self.params['indy_var']] = [indy_val]
                        point_data = aggregate_field_values(series_indy_var_val, point_data, self.params['line_type'])

                    # Calculate bootstrap results for the point
                    bootstrap_results = self._get_bootstrapped_stats(point_data, axis)
                    point_to_distrib[point] = bootstrap_results
                    n_stats = len(point_data)
                    logger.debug(f"Bootstrap results calculated for point {point_ind + 1}")

                else:
                    # Process derived points
                    logger.debug(f"Processing derived point: {point}")
                    bootstrap_results = self._get_bootstrapped_stats_for_derived(point, point_to_distrib, axis)
                    n_stats = 0

                # Save results to the output data frame
                out_frame.loc[point_ind, 'fcst_var'] = fcst_var
                out_frame.loc[point_ind, 'stat_value'] = bootstrap_results.value
                out_frame.loc[point_ind, 'stat_btcl'] = bootstrap_results.lower_bound
                out_frame.loc[point_ind, 'stat_btcu'] = bootstrap_results.upper_bound
                out_frame.loc[point_ind, 'nstats'] = n_stats
                logger.debug(f"Results saved for point {point_ind + 1}")

        else:
            logger.warning("Input data is empty. Returning an empty DataFrame.")
            out_frame = pd.DataFrame()

        logger.info("Completed stat calculations for axis '%s'", axis)
        return out_frame

    def calculate_stats_and_ci(self):
        """ Calculates aggregated statistics and confidants intervals
            ( if parameter num_iterations > 1) for each series point
            Writes output data to the file

        """
        logger = self.logger
        logger.info("Starting calculation of statistics and confidence intervals.")

        # Set random seed if present
        if self.params['random_seed'] is not None and self.params['random_seed'] != 'None':
            np.random.seed(self.params['random_seed'])
            logger.debug(f"Random seed set to: {self.params['random_seed']}")

        # Parse event equalization flag
        is_event_equal = parse_bool(self.params['event_equal'])
        logger.debug(f"Event equalization flag set to: {is_event_equal}")

        # Handle line type 'pct' by appending specific columns
        if self.params['line_type'] == 'pct':
            logger.debug("Adding additional columns for 'pct' line type.")
            self.column_names = np.append(self.column_names, ['T', 'oy_total', 'o_bar'])

        # Perform grouping for series_val_1
        logger.debug("Starting grouping for series_val_1.")
        series_val = self.params['series_val_1']
        group_to_value_index = 1
        if series_val:
            for key in series_val.keys():
                for val in series_val[key]:
                    if GROUP_SEPARATOR in val:
                        new_name = f'Group_y1_{group_to_value_index}'
                        self.group_to_value[new_name] = val
                        group_to_value_index += 1
                        logger.debug(f"Group created: {new_name} -> {val}")

        # Perform grouping for series_val_2
        logger.debug("Starting grouping for series_val_2.")
        series_val = self.params['series_val_2']
        if series_val:
            group_to_value_index = 1
            for key in series_val.keys():
                for val in series_val[key]:
                    if GROUP_SEPARATOR in val:
                        new_name = f'Group_y2_{group_to_value_index}'
                        self.group_to_value[new_name] = val
                        group_to_value_index += 1
                        logger.debug(f"Group created: {new_name} -> {val}")

        # Perform event equalization if required
        if is_event_equal:
            logger.debug("Performing event equalization.")
            self.input_data = perform_event_equalization(self.params, self.input_data)
            logger.info("Event equalization completed.")

        # Calculate statistics for axis 1
        logger.info("Calculating statistics for axis 1.")
        out_frame = self._proceed_with_axis("1")
        logger.debug(f"Axis 1 results shape: {out_frame.shape}")

        # Calculate statistics for axis 2 if needed
        if self.params['series_val_2']:
            logger.info("Calculating statistics for axis 2.")
            axis_2_frame = self._proceed_with_axis("2")
            out_frame = pd.concat([out_frame, axis_2_frame], ignore_index=True)
            logger.debug(f"Axis 2 results shape: {axis_2_frame.shape}")

        # Prepare to write the results to file
        header = True
        mode = 'w'
        if 'append_to_file' in self.params.keys() and self.params['append_to_file'] == 'True':
            header = False
            mode = 'a'
        logger.debug(f"Writing mode set to: {mode}, header: {header}")

        # Write the output to a CSV file
        output_file = self.params['agg_stat_output']
        logger.info(f"Writing results to file: {output_file}")
        try:
            out_frame.to_csv(output_file,
                            index=None, header=header, mode=mode,
                            sep="\t", na_rep="NA", float_format='%.' + str(PRECISION) + 'f')
            logger.info(f"Successfully wrote results to {output_file}")
        except Exception as e:
            logger.error(f"Error writing to file {output_file}: {e}", exc_info=True)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='List of agg_stat arguments')
    PARSER.add_argument("parameters_file", help="Path to YAML parameters file",
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    ARGS = PARSER.parse_args()
    with ARGS.parameters_file as parameters_file:
        PARAMS = yaml.load(parameters_file, Loader=yaml.FullLoader)

    AGG_STAT = AggStat(PARAMS)
    AGG_STAT.calculate_stats_and_ci()
