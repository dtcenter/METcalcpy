METcalcpy |version| Release Notes
_________________________________

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

Version |version| release notes (|release_date|)
------------------------------------------------

Version 1.0.0_beta4 release notes (20210302)

Bugfixes:

New Functionality:

* Support for GridDiag analysis (`#53 <https://github.com/dtcenter/METcalcpy/issues/53>`_)

* Calculation of CTC stats added: LODDS, ODDS, ORSS, SEDI, SEDS, EDI, EDS stats to the ctc_statistics module (`#60 <https://github.com/dtcenter/METcalcpy/issues/60>`_)

* Compute pairwise differences for "group" statistics (`#13 <https://github.com/dtcenter/METcalcpy/issues/13>`_)

Enhancements:

* Documentation added (`#6 <https://github.com/dtcenter/METcalcpy/issues/6>`_)

Internal:

* Initial design for data input logic to be used by METplotpy and METviewer (`#8 <https://github.com/dtcenter/METcalcpy/issues/8>`_) 

* UML diagram of design for data input logic (`#64 <https://github.com/dtcenter/METcalcpy/issues/64>`_) 

* Developer tests are working (`#65 <https://github.com/dtcenter/METcalcpy/issues/65>`_)


Version 1.0.0_beta3 release notes (20210127)

Bugfixes:

* Fix event equalization and agg_stat logic when the following fixed_vars_vals_input values are 'NA':
  fcst_thresh, fcst_thresh_1 (`#52 <https://github.com/dtcenter/METcalcpy/issues/52>`_)

* Fix the No-Skill reference line on Reliability Plot implemented in Python (`#26 <https://github.com/dtcenter/METcalcpy/issues/26>`_)

* Fix the calculation of ECNT_RMSE statistic so the Python and R implementations are consistent (`#42 <https://github.com/dtcenter/METcalcpy/issues/42>_`)

* Change the default setting of calc_difficulty_index to reproduce results originally generated from Naval Research Lab (`#37 <https://github.com/dtcenter/METcalcpy/issues/37>`_)

New Functionality:

* Add equivalence testing interval bounds to the existing continuous line type.  Two fields are added:
  lower bound and upper bound (similar to calculating a normal confidence interval except it requires
  using the non-central t-distribution (`#1 <https://github.com/dtcenter/METcalcpy/issues/1>`_)

* Enhance bootstrapping to support circular temporal block bootstrap with overlapping blocks (`#3 <https://github.com/dtcenter/METcalcpy/issues/3>`_)

* Add calculation for Difficulty Index from NRL (`#30 <https://github.com/dtcenter/METcalcpy/issues/30>`_)

* Create coordinate converter for [0,360] to [-180,180] (`#21 <https://github.com/dtcenter/METcalcpy/issues/21>`_)

* Add calculations used for Hovmoller Diagram, as contributed from the NOAA PSL diagnostics package (`#28 <https://github.com/dtcenter/METcalcpy/issues/28>`_)

* Create a method for creating series permutations that creates results consistent with R (`#44 <https://github.com/dtcenter/METcalcpy/issues/44>`_)

Enhancements:

* Replicate METviewer Reliability diagram using Python (`#48 <https://github.com/dtcenter/METcalcpy/issues/48>`_)

* Provide support for performing vertical interpolation of fields between grids with pressure and height coordinates (`#20 <https://github.com/dtcenter/METcalcpy/issues/20>`_)

* Incorporate the calculation of the Difficulty Index from Naval Research Lab  (`#27 <https://github.com/dtcenter/METcalcpy/issues/27>`_)

Internal:

* Confirm that Event Equalization for off-setting initialization is still working (`#16 <https://github.com/dtcenter/METcalcpy/issues/16>`_)

* Move convert_lons_indices() function from plot_blocking.py to utils.py in METcalcpy (`#33 <https://github.com/dtcenter/METcalcpy/issues/33>`_)

* Fix a typo (misspelling of package name) in code that creates packaging (`#43 <https://github.com/dtcenter/METcalcpy/issues/43>`_)

