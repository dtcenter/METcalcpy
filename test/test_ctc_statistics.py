import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch
from functools import cache
sys.path.append("../../")
from metcalcpy.util import ctc_statistics as ctc

cwd = os.path.dirname(__file__)

def test_asc_sort_by_ctc_fcst_thresh():
   """
      Test that the pandas dataframe is correctly sorting the
      fcst_thresh column in ascending order

   :return:

   """
   df = pd.read_csv(f"{cwd}/data/threshold.csv")
   sorted_df = ctc.sort_by_thresh(df)
   expected_fcst_thresh_list = ['NA','0', '>=0','>=0','>=0', '>0.01',  '<=1', '>=1&<=22.2', '>=1', '==3', '<5', '20', '>35&&<100.0', '>35', '100']
   expected_fcst_thresh = pd.Series(expected_fcst_thresh_list)
   sorted_fcst_thresh = sorted_df['fcst_thresh']
   assert(sorted_fcst_thresh.equals(other=expected_fcst_thresh))

def test_desc_sort_by_ctc_fcst_thresh():
   """
      Test that the pandas dataframe is correctly sorting the
      fcst_thresh column in descending order

   :return:
   """
   df = pd.read_csv(f"{cwd}/data/threshold.csv")
   sorted_df = ctc.sort_by_thresh(df, ascending=False)
   expected_fcst_thresh_list = ['100', '>35', '>35&&<100.0', '20', '<5','==3', '>=1','>=1&<=22.2', '<=1', '>0.01', '>=0', '>=0', '>=0', '0', 'NA']
   expected_fcst_thresh = pd.Series(expected_fcst_thresh_list)
   sorted_fcst_thresh = sorted_df['fcst_thresh']
   assert (sorted_fcst_thresh.equals(other=expected_fcst_thresh))



def test_calculate_ctc_roc_ascending():
    """
        Test that the created dataframe has appropriate values in the columns
        (requesting sorting to be done in ascending order).

        :return:
    """

    # read in the CTC input data
    df = pd.read_csv(f"{cwd}/data/ROC_CTC.data", sep='\t', header='infer')
    expected_pody_list = [0.8457663, 0.7634846, 0.5093934, 0.1228585]
    expected_pody = pd.Series(expected_pody_list)
    expected_thresh_list = ['>=1','>=2','>=3','>=4']
    expected_thresh = pd.Series(expected_thresh_list)
    ascending = True
    ctc_df = ctc.calculate_ctc_roc(df, ascending)
    thresh = ctc_df['thresh']
    pody = ctc_df['pody']

    desc_thresh = pd.Series(['>=4', '>=3', '>=2', '>=1'])

    # expect two pandas Series to have the same element and in the same order
    assert thresh.equals(other=expected_thresh)

    # Here are two pandas Series with the same items in different order.
    # This assertion is used to verify that anything out of order is False
    assert False == thresh.equals(other=desc_thresh)

    # This assert does not always work, depends on precision of machine
    # assert expected_pody.equals(other=pody), instead, take a position-by-position
    # difference of rounded values.
    for index, expected in enumerate(expected_pody):
        if ctc.round_half_up(expected) - ctc.round_half_up(pody[index]) == 0.0:
            pass
        else:
            assert False

    # if we get here, then all elements matched in value and position
    assert True

def test_calculate_ctc_roc_descending():
    """
        Test that the created dataframe has appropriate values in the columns
        (requesting sorting to be done in descending order).

        :return:
    """

    # read in the CTC input data
    df = pd.read_csv(f"{cwd}/data/ROC_CTC.data", sep='\t', header='infer')
    ascending = False
    expected_pody_list = [0.1228585,0.5093934,0.7634846,0.8457663]
    expected_pody = pd.Series(expected_pody_list)
    expected_thresh_list = ['>=4', '>=3', '>=2', '>=1']
    expected_thresh = pd.Series(expected_thresh_list)
    ctc_df = ctc.calculate_ctc_roc(df, ascending )
    thresh = ctc_df['thresh']
    pody = ctc_df['pody']

    assert thresh.equals(other=expected_thresh)

    # This assert does not always work, depends on precision of machine
    # assert pody.equals(other=expected_pody)
    for index, expected in enumerate(expected_pody):
        if ctc.round_half_up(expected) - ctc.round_half_up(pody[index]) == 0.0:
            pass
        else:
            assert False

    # if we get here, then all elements matched in value and position
    assert True

def test_CTC_ROC_thresh():
    # read in the CTC input data
    df = pd.read_csv(f"{cwd}/data/ROC_CTC_SFP.data", sep='\t', header='infer')
    ascending = False

    # All fcst_thresh values are >SFP30, so we expect only
    # one value returned for the threshold after calling calculate_ctc_roc()
    expected_thresh_list = ['>SFP30']
    expected_thresh = pd.Series(expected_thresh_list)
    ctc_df = ctc.calculate_ctc_roc(df, ascending)
    thresh = ctc_df['thresh']

    assert thresh.equals(other=expected_thresh)

    expected_pody_list = [0.8393175]
    expected_pody = pd.Series(expected_pody_list)
    pody = ctc_df['pody']

    # Use the round_half_up so we don't have inconsistent results due
    # to precision differences from host to host.
    for index, expected in enumerate(expected_pody):
        if ctc.round_half_up(expected) - ctc.round_half_up(pody[index]) == 0.0:
            pass
        else:
            assert False

    # if we get here, then all elements matched in value and position
    assert True

@cache
def ctc_data():
    df = pd.read_csv(f"{cwd}/data/ROC_CTC.data", sep='\t', header='infer')
    return df.to_numpy(), np.array(df.columns)


# Some of the functions here that
# are evaluating to None could be better 
# tested with a different dataset. e.g. acc
# needs a 'total' column in the test data.
@pytest.mark.parametrize(
    "func_name,expected",
    [
        ("fbias", 0.8549794),
        ("acc", None),
        ("fmean", None),
        ("pody", 0.5603757),
        ("pofd", 0.0368738),
        ("podn", 0.9631262),
        ("far", 0.3445741),
        ("csi", 0.432855),
        ("gss", None),
        ("hk", 0.5235019),
        ("hss", None),
        ("odds", 33.2937647),
        ("lodds", 3.5053691),
        ("bagss", None),
        ("eclv", None),
        ("ctc_total", None),
        ("cts_total", None),
        ("ctc_fn_on", 44984491.0),
        ("ctc_fn_oy", 2570049.0),
        ("ctc_fy_on", 1722257.0),
        ("ctc_fy_oy", 3275963.0),
        ("ctc_oy", 5846012.0),
        ("ctc_on", 46706748.0),
        ("ctc_fy", 46706748.0),
        ("ctc_fn", 47554540.0),
        ("odds1", 33.2937647),
        ("orss", 0.9416803),
        ("sedi", 0.7397156),
        ("seds", None),
        ("edi", 0.7014241),
        ("eds", None),
    ]
)
def test_ctc_statistics(func_name, expected):
    func_str = f"calculate_{func_name}"
    func = getattr(ctc, func_str)
    actual = func(*ctc_data())
    assert actual == expected

    # Check None is returned on Exception
    with patch.object(ctc, "sum_column_data_by_name", side_effect=TypeError):
        actual = func(*ctc_data())
    assert actual == None


    
if __name__ == "__main__":
    # test_asc_sort_by_ctc_fcst_thresh()
    test_desc_sort_by_ctc_fcst_thresh()
    # test_calculate_ctc_roc_ascending()
    # test_calculate_ctc_roc_descending()