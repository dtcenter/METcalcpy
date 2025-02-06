# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: nbrcnt_statistics.py
"""
import warnings

from metcalcpy.util.utils import round_half_up, sum_column_data_by_name, PRECISION, get_total_values
from metcalcpy.util.safe_log import safe_log

__author__ = 'Tatiana Burek'
__version__ = '0.1.0'


def calculate_nbr_fbs(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of NBR_FBS - Fractions Brier Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_FBS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Calculating total values from input data.")
        total = get_total_values(input_data, columns_names, aggregation)
        
        safe_log(logger, "debug", "Calculating Fractions Brier Score (FBS).")
        fbs = sum_column_data_by_name(input_data, columns_names, 'fbs') / total
        
        safe_log(logger, "debug", "Rounding the FBS to the defined precision.")
        result = round_half_up(fbs, PRECISION)
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_FBS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_fss(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of NBR_FSS - Fractions Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_FSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Calculating total values from input data.")
        total = get_total_values(input_data, columns_names, aggregation)
        
        safe_log(logger, "debug", "Calculating denominator for FSS.")
        fss_den = sum_column_data_by_name(input_data, columns_names, 'fss') / total
        
        safe_log(logger, "debug", "Calculating Fractions Brier Score (FBS).")
        fbs = sum_column_data_by_name(input_data, columns_names, 'fbs') / total
        
        safe_log(logger, "debug", "Calculating Fractions Skill Score (FSS).")
        fss = 1.0 - fbs / fss_den
        
        safe_log(logger, "debug", "Rounding the FSS to the defined precision.")
        result = round_half_up(fss, PRECISION)
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_FSS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_afss(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of NBR_AFSS - Asymptotic Fractions Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_AFSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Calculating total values from input data.")
        total = get_total_values(input_data, columns_names, aggregation)
        
        safe_log(logger, "debug", "Calculating forecast rate (f_rate).")
        f_rate = sum_column_data_by_name(input_data, columns_names, 'f_rate') / total
        
        safe_log(logger, "debug", "Calculating observation rate (o_rate).")
        o_rate = sum_column_data_by_name(input_data, columns_names, 'o_rate') / total
        
        safe_log(logger, "debug", "Calculating numerator for AFSS.")
        afss_num = 2.0 * f_rate * o_rate
        
        safe_log(logger, "debug", "Calculating denominator for AFSS.")
        afss_den = f_rate * f_rate + o_rate * o_rate
        
        safe_log(logger, "debug", "Calculating Asymptotic Fractions Skill Score (AFSS).")
        afss = afss_num / afss_den
        
        safe_log(logger, "debug", "Rounding the AFSS to the defined precision.")
        result = round_half_up(afss, PRECISION)
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_AFSS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_ufss(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of NBR_UFSS - Uniform Fractions Skill Score

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_UFSS as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Calculating total values from input data.")
        total = get_total_values(input_data, columns_names, aggregation)
        
        safe_log(logger, "debug", "Calculating observation rate (o_rate).")
        o_rate = sum_column_data_by_name(input_data, columns_names, 'o_rate') / total
        
        safe_log(logger, "debug", "Calculating Uniform Fractions Skill Score (UFSS).")
        ufss = 0.5 + o_rate / 2.0
        
        safe_log(logger, "debug", "Rounding the UFSS to the defined precision.")
        result = round_half_up(ufss, PRECISION)
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_UFSS calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_f_rate(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of NBR_F_RATE - Forecast event frequency

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_F_RATE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Calculating total values from input data.")
        total = get_total_values(input_data, columns_names, aggregation)
        
        safe_log(logger, "debug", "Calculating forecast event frequency (f_rate).")
        f_rate = sum_column_data_by_name(input_data, columns_names, 'f_rate') / total
        
        safe_log(logger, "debug", "Rounding the F_RATE to the defined precision.")
        result = round_half_up(f_rate, PRECISION)
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_F_RATE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_o_rate(input_data, columns_names, aggregation=False, logger=None):
    """Performs calculation of NBR_O_RATE - Observed event frequency

        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array
            aggregation: if the aggregation on fields was performed

        Returns:
            calculated NBR_O_RATE as float
            or None if some of the data values are missing or invalid
    """
    warnings.filterwarnings('error')
    try:
        safe_log(logger, "debug", "Calculating total values from input data.")
        total = get_total_values(input_data, columns_names, aggregation)
        
        safe_log(logger, "debug", "Calculating observed event frequency (o_rate).")
        o_rate = sum_column_data_by_name(input_data, columns_names, 'o_rate') / total
        
        safe_log(logger, "debug", "Rounding the O_RATE to the defined precision.")
        result = round_half_up(o_rate, PRECISION)
    
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during NBR_O_RATE calculation: {str(e)}")
        result = None
    warnings.filterwarnings('ignore')
    return result


def calculate_nbr_cnt_total(input_data, columns_names, logger=None):
    """Performs calculation of Total number of matched pairs for Neighborhood Continuous Statistics
        Args:
            input_data: 2-dimensional numpy array with data for the calculation
                1st dimension - the row of data frame
                2nd dimension - the column of data frame
            columns_names: names of the columns for the 2nd dimension as Numpy array

        Returns:
            calculated Total number of matched pairs as float
            or None if some of the data values are missing or invalid
    """
    try:
        safe_log(logger, "debug", "Starting calculation of total matched pairs.")
        total = sum_column_data_by_name(input_data, columns_names, 'total')
        
        safe_log(logger, "debug", f"Total matched pairs before rounding: {total}")
        result = round_half_up(total, PRECISION)
        
        safe_log(logger, "debug", f"Total matched pairs after rounding: {result}")
    except (TypeError, ZeroDivisionError, Warning, ValueError) as e:
        safe_log(logger, "warning", f"Exception occurred during total matched pairs calculation: {str(e)}")
        result = None
