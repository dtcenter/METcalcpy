METcalcpy |version| Release Notes
_________________________________

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

Version |version| release notes (|release_date|)
------------------------------------------------

Version 1.1.0 release notes (20210511)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

Version 1.0.0 release notes (20210511)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New Functionality:

* Add calculations for Space-time Coherence Spectra from NOAA PSL Diagnostics Package (`#29 <https://github.com/dtcenter/METcalcpy/issues/29>`_)

* Utilize METdbload .stat file reader utilities to allow METcalcpy to read these files (`#31 <https://github.com/dtcenter/METcalcpy/issues/31>`_)

* Utilize METdbload .stat file reader utilities to allow METcalcpy to read these files (`#31 <https://github.com/dtcenter/METcalcpy/issues/31>`_)

* Create a template for reading in a data file (of raw data) and create a data file reader (`#7 <https://github.com/dtcenter/METcalcpy/issues/7>`_)

* Add new ECNT stats to aggregation logic (`#69 <https://github.com/dtcenter/METcalcpy/issues/69>`_)

* Implement a method that returns specified column values (`#73 <https://github.com/dtcenter/METcalcpy/issues/73>`_)

* Support for GridDiag analysis (`#53 <https://github.com/dtcenter/METcalcpy/issues/53>`_)

* Calculation of CTC stats added: LODDS, ODDS, ORSS, SEDI, SEDS, EDI, EDS stats to the ctc_statistics module** (`#60 <https://github.com/dtcenter/METcalcpy/issues/60>`_)

* Compute pairwise differences for "group" statistics (`#13 <https://github.com/dtcenter/METcalcpy/issues/13>`_)

* Add equivalence testing interval bounds to the existing continuous line type. Two fields are added:lower bound and upper bound (similar to calculating a normal confidence interval except it requires using the non-central t-distribution (`#1 <https://github.com/dtcenter/METcalcpy/issues/1>`_)

* Add calculation for Difficulty Index from NRL (`#88 <https://github.com/dtcenter/METplotpy/issues/88>`_)

* Enhance bootstrapping to support circular temporal block bootstrap with overlapping blocks (`#3 <https://github.com/dtcenter/METcalcpy/issues/3>`_)

* Add equivalence testing interval bounds to the existing continuous line type.  Two fields are added:
    lower bound and upper bound (similar to calculating a normal confidence interval except it requires
    using the non-central t-distribution (`#1 <https://github.com/dtcenter/METcalcpy/issues/1>`_)

* Add calculation for Difficulty Index from NRL (`#30 <https://github.com/dtcenter/METcalcpy/issues/30>`_)

* Create coordinate converter for [0,360] to [-180,180] (`#21 <https://github.com/dtcenter/METcalcpy/issues/21>`_)

* Add calculations used for Hovmoller Diagram, as contributed from the NOAA PSL diagnostics package (`#28 <https://github.com/dtcenter/METcalcpy/issues/28>`_)

* Create a method for creating series permutations that creates results consistent with R (`#44 <https://github.com/dtcenter/METcalcpy/issues/44>`_)

* Add calculation for Difficulty Index from NRL** (`#30 <https://github.com/dtcenter/METcalcpy/issues/30>`_)

* Create coordinate converter for [0,360] to [-180,180] (`#21 <https://github.com/dtcenter/METcalcpy/issues/21>`_)

Enhancements:

* Documentation added (`#6 <https://github.com/dtcenter/METcalcpy/issues/6>`_)

* Replicate METviewer Reliability diagram using Python (`#48 <https://github.com/dtcenter/METcalcpy/issues/48>`_)

* Provide support for performing vertical interpolation of fields between grids with pressure and height coordinates (`#20 <https://github.com/dtcenter/METcalcpy/issues/20>`_)

* Incorporate the calculation of the Difficulty Index from Naval Research Lab  (`#27 <https://github.com/dtcenter/METcalcpy/issues/27>`_)

Internal:

* Rename boot Confidence Interval's (CI) columns (`#79 <https://github.com/dtcenter/METcalcpy/issues/79>`_)

* Initial design for data input logic to be used by METplotpy and METviewer (`#8 <https://github.com/dtcenter/METcalcpy/issues/8>`_)

* UML diagram of design for data input logic (`#64 <https://github.com/dtcenter/METcalcpy/issues/64>`_)

* Developer tests are working (`#65 <https://github.com/dtcenter/METcalcpy/issues/65>`_)

*  Fix typo in setup.py (`#43 <https://github.com/dtcenter/METcalcpy/issues/43>`_)

* Confirm that Event Equalization for off-setting initialization is still working (`#16 <https://github.com/dtcenter/METcalcpy/issues/16>`_)

* Move convert_lons_indices() function from plot_blocking.py to utils.py in METcalcpy (`#33 <https://github.com/dtcenter/METcalcpy/issues/33>`_)

* Fix a typo (misspelling of package name) in code that creates packaging (`#43 <https://github.com/dtcenter/METcalcpy/issues/43>`_)

* Move convert_lons_indices() function from plot_blocking.py to utils.py in METcalcpy (`#33 <https://github.com/dtcenter/METcalcpy/issues/33>`_)

Bugfixes:

* Fix event equalization and agg_stat logic (`#52 <https://github.com/dtcenter/METcalcpy/issues/52>`_)

* Fix the No-Skill reference line on Reliability Plot implemented in Python (`#26 <https://github.com/dtcenter/METcalcpy/issues/26>`_)

* Fix the calculation of ECNT_RMSE statistic so the Python and R implementations are consistent (`#42 <https://github.com/dtcenter/METcalcpy/issues/42>`_)

* Change the default setting of calc_difficulty_index to reproduce results originally generated from Naval Research Lab (`#37 <https://github.com/dtcenter/METcalcpy/issues/37>`_)

* Fix the No-Skill reference line on Reliability Plot implemented in Python (`#26 <https://github.com/dtcenter/METcalcpy/issues/26>`_)

* Fix the calculation of ECNT_RMSE statistic so the Python and R implementations are consistent (`#42 <https://github.com/dtcenter/METcalcpy/issues/42>`_)

* Change the default setting of calc_difficulty_index to reproduce results originally generated from Naval Research Lab (`#37 <https://github.com/dtcenter/METcalcpy/issues/37>`_) 

* Fix event equalization and agg_stat logic (`#52 <https://github.com/dtcenter/METcalcpy/issues/52>`_)

