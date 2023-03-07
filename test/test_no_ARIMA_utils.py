import numpy as np
import pandas as pd
import metcalcpy.util.utils as utils



def test_no_arima():
    data_file = "./data/scorecard.csv"
    df = pd.read_csv(data_file)

    stat_values:pd.Series = df['stat_value']
    # convert dataframe to numpy array
    np_data:np.array = stat_values.to_numpy()
    size_data_from_file = np_data.size

    try:
       std_err, ratio_flag, ar_1, size_data =utils.compute_std_err_from_median_variance_inflation_factor(np_data)
       assert size_data_from_file == size_data
    except NameError:
        # if ARIMA is still present, expect "NameError: name 'ARIMA' is not defined
        assert False

