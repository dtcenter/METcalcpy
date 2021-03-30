import pytest
import pandas as pd
import sys
sys.path.append("../../")
from metcalcpy.util import ctc_statistics as ctc

def test_asc_sort_by_ctc_fcst_thresh():
   """
      Test that the pandas dataframe is correctly sorting the
      fcst_thresh column in ascending order

   :return:

   """
   df = pd.read_csv("./data/threshold.csv")
   sorted_df = ctc.sort_by_thresh(df)
   expected_fcst_thresh_list = ['0', '>=0','>=0','>=0','>0.01', '<=1', '>=1', '==3', '<5', '20', '>35', '100']
   expected_fcst_thresh = pd.Series(expected_fcst_thresh_list)
   sorted_fcst_thresh = sorted_df['fcst_thresh']
   assert(sorted_fcst_thresh.equals(other=expected_fcst_thresh))


def test_desc_sort_by_ctc_fcst_thresh():
   """
      Test that the pandas dataframe is correctly sorting the
      fcst_thresh column in descending order

   :return:
   """
   df = pd.read_csv("./data/threshold.csv")
   sorted_df = ctc.sort_by_thresh(df, ascending=False)
   expected_fcst_thresh_list = ['100', '>35', '20', '<5','==3', '>=1', '<=1', '>0.01', '>=0', '>=0', '>=0', '0']
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
    df = pd.read_csv("./data/ROC_CTC.data", sep='\t', header='infer')
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
    df = pd.read_csv("./data/ROC_CTC.data", sep='\t', header='infer')
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
    df = pd.read_csv("./data/ROC_CTC_SFP.data", sep='\t', header='infer')
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


    
if __name__ == "__main__":
    # test_asc_sort_by_ctc_fcst_thresh()
    test_desc_sort_by_ctc_fcst_thresh()
    # test_calculate_ctc_roc_ascending()
    # test_calculate_ctc_roc_descending()