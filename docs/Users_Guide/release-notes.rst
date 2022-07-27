METcalcpy |version| Release Notes
_________________________________

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

Version |version| release notes (|release_date|)
------------------------------------------------

Version 1.1.1 release notes (20220727)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bug Fix:

* Automatically reorder the data so regimes match between forecast and observations

Version 1.1.0 release notes (20220311)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New Functionality:

* Add OMI and RMM statistics (`#89 <https://github.com/dtcenter/METcalcpy/issues/89>`_)

* Add calculation of CTC statistics (`#77 <https://github.com/dtcenter/METcalcpy/issues/77>`_)

* Add sorting of CTC dataframe by fcst_thresh (`#75 <https://github.com/dtcenter/METcalcpy/issues/75>`_)

* Add calculation of the Scatter Index statistic (`#108 <https://github.com/dtcenter/METcalcpy/issues/108>`_)

* **Add Zonal and Meridional mean calculation from basic.py** (`#126 <https://github.com/dtcenter/METcalcpy/issues/126>`_)

* **Add supporting functionality for ECLV plot** (`#128 <https://github.com/dtcenter/METcalcpy/issues/128>`_)


Enhancements:
 
* Add vertical_interp support for multiple pressure coordinate systems (`#63 <https://github.com/dtcenter/METcalcpy/issues/63>`_)

* Change ',' as a separator for the series group to ':' (`#117 <https://github.com/dtcenter/METcalcpy/issues/117>`_)

* Enhance METcalcpy to aggregate and plot the HSS_EC statistic from the MCTS line type (`#107 <https://github.com/dtcenter/METcalcpy/issues/107>`_)

* **Reorganize the METcalcpy directory structure to separate the statistics modules from the pre-processing, diagnostics, and util modules** (`#125 <https://github.com/dtcenter/METcalcpy/issues/125>`_)

Internal:

* Add bootstrap package classes (`#96 <https://github.com/dtcenter/METcalcpy/issues/96>`_)

* Add pingouin package classes (`#98 <https://github.com/dtcenter/METcalcpy/issues/98>`_)

* Update documentation to reference GitHub Discussions instead of MET Help (`#100 <https://github.com/dtcenter/METcalcpy/issues/100>`_)

* Implement Auto- and Cross- Covariance and -Correlation Function Estimation function for Revision series for MODE-TD (`#121 <https://github.com/dtcenter/METcalcpy/issues/121>`_)

* Add copyright information to all Python source code (`#150 <https://github.com/dtcenter/METcalcpy/issues/150>`_)


Bugfixes:

* Fixed CTC statistics for the ROC diagram (`#77 <https://github.com/dtcenter/METcalcpy/issues/77>`_)

* **update input parameters for test_agg_stats_with_groups script with GROUP_SEPARATOR** (`#138 <https://github.com/dtcenter/METcalcpy/issues/138>`_)

* **permutations are not created if the list is passed as a parameter** (`#136 <https://github.com/dtcenter/METcalcpy/issues/136>`_)

* plots with groups with date values don't get created (`#122 <https://github.com/dtcenter/METcalcpy/issues/122>`_)

* Prepare data for a line plot with different forecast variables plotted on y1 and y2 axis (`#113 <https://github.com/dtcenter/METcalcpy/issues/113>`_)

* Histogram plots can't be generated due to incorrect sys.modules path (`#209 <https://github.com/dtcenter/METcalcpy/issues/209>`_)

