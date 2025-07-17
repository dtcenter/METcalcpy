
*****************************
METcalcpy Release Information
*****************************

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

METcalcpy Release Notes
=======================

METcalcpy Version 3.1.0 release notes (20250716)
----------------------------------------------------


.. dropdown:: Bugfixes

   * Initialize stat_vals list in agg_stat_bootstrap code (`#420 <https://github.com/dtcenter/METcalcpy/issues/420>`_) 
	      
.. dropdown:: Documentation

   * Enhance the Table of Contents to include all METplus components (`#432 <https://github.com/dtcenter/METcalcpy/pull/432>`_)
   * Update the Code Support Section (`#443 <https://github.com/dtcenter/METcalcpy/issues/443>`_)

.. dropdown:: Enhancements

   * **Add CTP and HI functions as new diagnostics in METcalcpy** (`#378 <https://github.com/dtcenter/METcalcpy/issues/378>`_)

.. dropdown:: Repository, build, and test

   * Migrate install to pyproject.toml (`#414 <https://github.com/dtcenter/METcalcpy/issues/414>`_)
   * Add basic tests for untested modules (`#415 <https://github.com/dtcenter/METcalcpy/issues/415>`_)
   * Add tests for mode_3d_volrat_statistics.py (`#417 <https://github.com/dtcenter/METcalcpy/issues/417>`_)
   * Update code after upgrading the numpy version (`#425 <https://github.com/dtcenter/METcalcpy/pull/425>`_)
   * Update code after upgrading the xarray version (`#431 <https://github.com/dtcenter/METcalcpy/pull/431>`_)  
   * Update infrastructure to reflect move to developing with Python 3.12 (`#430 <https://github.com/dtcenter/METcalcpy/pull/430>`_)
   * Update modulefiles used on various machines (`#427 <https://github.com/dtcenter/METcalcpy/issues/427>`_)     
   * Update installation modulefiles for Python 3.12 (`#436 <https://github.com/dtcenter/METcalcpy/issues/436>`_)



METcalcpy Upgrade Instructions
==============================

.. note::

   In the METcalcpy-3.1.0-beta2 release, METcalcpy switched from development
   with Python 3.10.4 to development with Python 3.12. View the
   requirements.txt/nco_requirements.txt file at the top level of the
   repository for version numbers for the corresponding third-party packages.
   
