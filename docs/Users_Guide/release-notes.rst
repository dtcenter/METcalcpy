*****************************
METcalcpy Release Information
*****************************

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

METcalcpy Release Notes
=======================

METcalcpy Version 2.0.2 release notes (20230223)
------------------------------------------------

* Bugfixes:


   * **Address Warnings from pandas in METcalcpy code** 
     (`#249 <https://github.com/dtcenter/METcalcpy/issues/249>`_)

METcalcpy Version 2.0.1 release notes (20230125)
------------------------------------------------
* New Functionality:

* Enhancements:

* Internal:

* Bugfixes:

   * **Add nco_requirements.txt file** 
     (`#263 <https://github.com/dtcenter/METcalcpy/issues/263>`_)

   * **Fix test_scorecard.py with pandas-1.5.1** 
     (`#251 <https://github.com/dtcenter/METcalcpy/issues/251>`_)

   * **Fix test_event_equalize.py** 
     (`#250 <https://github.com/dtcenter/METcalcpy/issues/250>`_)


METcalcpy Version 2.0.0 release notes (20221207)
------------------------------------------------
* New Functionality:

   * **Add Zonal and Meridional Mean calculations** 
     (`#133 <https://github.com/dtcenter/METcalcpy/issues/133>`_)

   * **Implement aggregation of CRPS_EMP_FAIR across multiple cases** 
     (`#215 <https://github.com/dtcenter/METcalcpy/issues/215>`_)

   * **Implement aggregation of ECNT statistics** 
     (`#229 <https://github.com/dtcenter/METcalcpy/issues/229>`_)

   * **Add CRPS, ECNT_CRPS, ECNT_RMSE, ECNT_ME to perfect scores statistics groups** 
     (`#218 <https://github.com/dtcenter/METcalcpy/issues/218>`_)

   * **Add EC_VALUE to aggregation of CTC and CTS values** (`#198 <https://github.com/dtcenter/METcalcpy/issues/198>`_)

   * **Add in the new scripts for the MJO ENSO use case** (`#207 <https://github.com/dtcenter/METcalcpy/issues/207>`_)

   * **Add summary and aggregation logic for calculating VCNT ANOM_CORR_UNCNTR and ANOM_CORR** (`#200 <https://github.com/dtcenter/METcalcpy/issues/200>`_)

* Enhancements:

   * **Support calculation of Revision Series Data**  (`#181 <https://github.com/dtcenter/METcalcpy/issues/181>`_)

   * **Create a Contributor's Guide** (`#178 <https://github.com/dtcenter/METcalcpy/issues/178>`_)

   * **Setup SonarQube** (`#37 <https://github.com/dtcenter/METcalcpy/issues/37>`_)


* Internal:

   * **Remove statsmodels and patsy from METplus and analysis tools** 
     (`#219 <https://github.com/dtcenter/METcalcpy/issues/219>`_)

   * **Fix github Actions warnings** 
     (`#218 <https://github.com/dtcenter/METcalcpy/issues/218>`_)

   * **Create checksum for released code** (`#209 <https://github.com/dtcenter/METcalcpy/issues/209>`_)

   * **Add modulefiles used for installation on various machines** (`#204 <https://github.com/dtcenter/METcalcpy/issues/204>`_)

   * **Identify minimal "bare-bones" Python packages to inform operational installation** (`#152 <https://github.com/dtcenter/METcalcpy/issues/152>`_)

   * **Convert scorecard.R_tmp to Python** (`#179 <https://github.com/dtcenter/METcalcpy/issues/179>`_)

* Bugfixes:

   * **Vertical Interpolation DimensionalityError in migrating from Python 3.7.10 to Python 3.8.6** (`#180 <https://github.com/dtcenter/METcalcpy/issues/180>`_)

   * **Address Github Dependabot Issues** (`#193 <https://github.com/dtcenter/METcalcpy/issues/193>`_)

   * **Deprecation and other warnings in event_equalize.py and other modules** (`#153 <https://github.com/dtcenter/METcalcpy/issues/153>`_)

   * **Some METcalcpy tests fail with Python 3.8 and upgraded packages** (`#154 <https://github.com/dtcenter/METcalcpy/issues/154>`_)
    
METcalcpy Upgrade Instructions
==============================

Upgrade instructions will be listed here if they are applicable
for this release.
