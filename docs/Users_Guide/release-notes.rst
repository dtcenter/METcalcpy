METcalcpy |version| Release Notes
_________________________________

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

Version |version| release notes (|release_date|)
------------------------------------------------

Version 1.1.0 release notes (20220218)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

beta6 release
^^^^^^^^^^^^^

New Functionality:

Enhancements:
 
* **Reorganize the METcalcpy directory structure to separate the statistics modules from the pre-processing, diagnostics, and util modules ('#125 <https://github.com/dtcenter/METcalcpy/issues/125>'_)** 


Internal:


Bugfixes:


Version 1.1.0 release notes (20220119)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

beta5 release
^^^^^^^^^^^^^


New Functionality:

* **Add Zonal and Meridional mean calculation from basic.py** (`#126 <https://github.com/dtcenter/METcalcpy/issues/126>`_)

* **Add supporting functionality for ECLV plot** (`#128 <https://github.com/dtcenter/METcalcpy/issues/128>`_)


Enhancements:



Internal:



Bugfixes:

* **update input parameters for test_agg_stats_with_groups script with GROUP_SEPARATOR** (`#138 <https://github.com/dtcenter/METcalcpy/issues/138>`_)


* **permutations are not created if the list is passed as a parameter** (`#136 <https://github.com/dtcenter/METcalcpy/issues/136>`_)


beta4 release
^^^^^^^^^^^^^

New Functionality:

Enhancements:


Internal:


* Implement Auto- and Cross- Covariance and -Correlation Function Estimation function for Revision series for MODE-TD (`#121 <https://github.com/dtcenter/METcalcpy/issues/121>`_)

Bugfixes:

* plots with groups with date values don't get created (`#122 <https://github.com/dtcenter/METcalcpy/issues/122>`_)


beta3 release
^^^^^^^^^^^^^


New Functionality:

Enhancements:

* Add vertical_interp support for multiple pressure coordinate systems (`#63 <https://github.com/dtcenter/METcalcpy/issues/63>`_)

* Change ',' as a separator for the series group to ':' (`#117 <https://github.com/dtcenter/METcalcpy/issues/117>`_)


Internal:


Bugfixes:

* Prepare data for a line plot with different forecast variables plotted on y1 and y2 axis (`#113 <https://github.com/dtcenter/METcalcpy/issues/113>`_)



beta2 release
^^^^^^^^^^^^^

New Functionality:

* Add calculation of the Scatter Index statistic (`#108 <https://github.com/dtcenter/METcalcpy/issues/108>`_)



Enhancements:

* Enhance METcalcpy to aggregate and plot the HSS_EC statistic from the MCTS line type (`#107 <https://github.com/dtcenter/METcalcpy/issues/107>`_)


Internal:

* Update documentation to reference GitHub Discussions instead of MET Help (`#100 <https://github.com/dtcenter/METcalcpy/issues/100>`_)

Bugfixes:




beta1 release
^^^^^^^^^^^^^

New Functionality:

* Add OMI and RMM statistics (`#89 <https://github.com/dtcenter/METcalcpy/issues/89>`_)

* Add calculation of CTC statistics (`#77 <https://github.com/dtcenter/METcalcpy/issues/77>`_)

* Add sorting of CTC dataframe by fcst_thresh (`#75 <https://github.com/dtcenter/METcalcpy/issues/75>`_)
 
Enhancements:

Internal:

* Add bootstrap package classes (`#96 <https://github.com/dtcenter/METcalcpy/issues/96>`_)

* Add pingouin package classes (`#98 <https://github.com/dtcenter/METcalcpy/issues/98>`_)

Bugfixes:

* Fixed CTC statistics for the ROC diagram (`#77 <https://github.com/dtcenter/METcalcpy/issues/77>`_)

