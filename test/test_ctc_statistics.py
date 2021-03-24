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
   sorted_df = ctc.sort_by_ctc_fcst_thresh(df)
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
   sorted_df = ctc.sort_by_ctc_fcst_thresh(df, ascending=False)
   expected_fcst_thresh_list = ['100', '>35', '20', '<5','==3', '>=1', '<=1', '>0.01', '>=0', '>=0', '>=0', '0']
   expected_fcst_thresh = pd.Series(expected_fcst_thresh_list)
   sorted_fcst_thresh = sorted_df['fcst_thresh']
   assert (sorted_fcst_thresh.equals(other=expected_fcst_thresh))


if __name__ == "__main__":
    test_asc_sort_by_ctc_fcst_thresh()
    test_desc_sort_by_ctc_fcst_thresh()