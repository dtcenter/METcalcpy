*****************************
METcalcpy Release Information
*****************************

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

METcalcpy Release Notes
=======================

METcalcpy Version 3.1.0-beta2 release notes (20250326)
------------------------------------------------------

.. dropdown:: Bugfixes

   * Initialize stat_vals list in agg_stat_bootstrap code (`#420 <https://github.com/dtcenter/METcalcpy/issues/420>`_) 
	      
.. dropdown:: Documentation

   * Enhance the Table of Contents to include all METplus components (`#432 <https://github.com/dtcenter/METcalcpy/pull/432>`_)  

.. dropdown:: Repository, build, and test

   * Update code after upgrading the numpy version (`#425 <https://github.com/dtcenter/METcalcpy/pull/425>`_)
   * Update code after upgrading the xarray version (`#431 <https://github.com/dtcenter/METcalcpy/pull/431>`_)  
   * Update infrastructure to reflect move to developing with Python 3.12 (`#430 <https://github.com/dtcenter/METcalcpy/pull/430>`_)
   * Update modulefiles used on various machines (`#427 <https://github.com/dtcenter/METcalcpy/issues/427>`_)     
		

METcalcpy Version 3.1.0-beta1 release notes (20250122)
------------------------------------------------------

.. dropdown:: Repository, build, and test

  * Migrate install to pyproject.toml (`#414 <https://github.com/dtcenter/METcalcpy/issues/414>`_)
  * Add basic tests for untested modules (`#415 <https://github.com/dtcenter/METcalcpy/issues/415>`_)
  * 3d volrat tests (`#417 <https://github.com/dtcenter/METcalcpy/issues/417>`_)


METcalcpy Upgrade Instructions
==============================

.. note::

   In the METcalcpy-3.1.0-beta2 release, METcalcpy switched from developing
   with Python 3.10.4 to developing with Python 3.12.
