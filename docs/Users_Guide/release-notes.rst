
** New in v1.0**


Bugfixes:

* Fix the No-Skill reference line on Reliability Plot implemented in Python (`#26 <https://github.com/dtcenter/METcalcpy/issues/26>`_)

* Fix the calculation of ECNT_RMSE statistic so the Python and R implementations are consistent (`#42 <https://github.com/dtcenter/METcalcpy/issues/42>_`) 

* Change the default setting of calc_difficulty_index to reproduce results originally generated from Naval Research Lab (`#37 <https://github.com/dtcenter/METcalcpy/issues/37>`_)
New Functionality:

* Add calculation for Difficulty Index from NRL (`#30 <https://github.com/dtcenter/METcalcpy/issues/30>`_)

* Create coordinate converter for [0,360] to [-180,180] (`#21 <https://github.com/dtcenter/METcalcpy/issues/21>`_) 
 
* Add calculations used for Hovmoller Diagram, as contributed from the NOAA PSL diagnostics package (`#28 <https://github.com/dtcenter/METcalcpy/issues/28>`_)

* Create a method for creating series permutations that creates results consistent with R (`#44 <https://github.com/dtcenter/METcalcpy/issues/44>`_)

Enhancements:

* Replicate METviewer Reliability diagram using Python (`#48 <https://github.com/dtcenter/METcalcpy/issues/48>`_)

* Provide support for performing vertical interpolation of fields between grids with pressure and height coordinates (`#20 <https://github.com/dtcenter/METcalcpy/issues/20>`_)

* Incorporate the calculation of the Difficulty Index from Naval Research Lab  (`#27 <https://github.com/dtcenter/METcalcpy/issues/27>`_)

Internal:

* Move convert_lons_indices() function from plot_blocking.py to utils.py in METcalcpy (`#33 <https://github.com/dtcenter/METcalcpy/issues/33>`_)

* Fix a typo (misspelling of package name) in code that creates packaging (`#43 <https://github.com/dtcenter/METcalcpy/issues/43>`_)
