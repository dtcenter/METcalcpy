*****************************
METcalcpy Release Information
*****************************

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature: `METcalcpy GitHub issues. <https://github.com/dtcenter/METcalcpy/issues>`_

METcalcpy Release Notes
=======================

METcalcpy Velsion 3.0.0-beta6 release notes (20241017)
------------------------------------------------------

  .. dropdown:: New Functionality

     None

  .. dropdown:: Enhancements

     * Improve logging for 5 STIGs (`METplus-Internal#46 <https://github.com/dtcenter/METplus-Internal/issues/46>`_)

  .. dropdown:: Internal

     None

  .. dropdown:: Bugfixes

     * Bugfix: MODE CSI calculations result in spurious results (`#360 <https://github.com/dtcenter/METcalcpy/issues/360>`_)


METcalcpy Velsion 3.0.0-beta5 release notes (20240628)
------------------------------------------------------


  .. dropdown:: New Functionality

     * **Add updates to MPR writer and fix bugs for stratosphere** (`#385 <https://github.com/dtcenter/METcalcpy/issues/385>`_)

  .. dropdown:: Enhancements

     * **Enhance METcalcpy to use the TOTAL_DIR column when aggregate statistics wind direction statistics in the VL1L2, VAL1L2, and VCNT columns** (`#384 <https://github.com/dtcenter/METcalcpy/issues/384>`_)

  .. dropdown:: Internal

     * Update GitHub issue and pull request templates to reflect the current development workflow details  (`#326 <https://github.com/dtcenter/METcalcpy/issues/326>`_)
     * Consider using only .yml or only .yaml extensions  (`#349 <https://github.com/dtcenter/METcalcpy/issues/349>`_)
     * Code coverage statistics  (`#54 <https://github.com/dtcenter/METplus-Internal/issues/54>`_)


  .. dropdown:: Bugfixes


METcalcpy Velsion 3.0.0-beta4 release notes (20240417)
------------------------------------------------------


  .. dropdown:: New Functionality


  .. dropdown:: Enhancements
 
     * Add calculation for Terrestrial Coupling Index (`#364 <https://github.com/dtcenter/METcalcpy/issues/364>`_)
     * Enhance aggregate statistics for ECNT,VL1L2,VAL1L2 and VCNT (`#361 <https://github.com/dtcenter/METcalcpy/issues/361>`_)


  .. dropdown:: Internal

     * Develop sonarqube capabilities  (`#367 <https://github.com/dtcenter/METcalcpy/issues/367>`_)
     * Add github action for sonarqube   (`#366 <https://github.com/dtcenter/METcalcpy/issues/366>`_)
     * Updated pythoh requirements.txt   (`#355 <https://github.com/dtcenter/METcalcpy/issues/355>`_)
     * Modified python requirements section of Users Guide   (`#352 <https://github.com/dtcenter/METcalcpy/issues/352>`_)


  .. dropdown:: Bugfixes


     * Address negative values returned by calculate_bcmse() and calculate_bcrmse() in sl1l2_statistics module (`#329 <https://github.com/dtcenter/METcalcpy/issues/329>`_)

METcalcpy Velsion 3.0.0-beta3 release notes (20240207)
------------------------------------------------------


  .. dropdown:: New Functionality


  .. dropdown:: Enhancements
 
     * **Create aggregation support for MET .stat output** (`#325 <https://github.com/dtcenter/METcalcpy/issues/325>`_)


  .. dropdown:: Internal

     * Update GitHub actions workflows to switch from node 16 to node 20  (`#345 <https://github.com/dtcenter/METcalcpy/issues/345>`_)


  .. dropdown:: Bugfixes


     * Address negative values returned by calculate_bcmse() and calculate_bcrmse() in sl1l2_statistics module (`#329 <https://github.com/dtcenter/METcalcpy/issues/329>`_)


METcalcpy Velsion 3.0.0-beta2 release notes (20231114)
------------------------------------------------------

  .. dropdown:: New Functionality

  .. dropdown:: Enhancements

  .. dropdown:: Internal

     * Change second person references to third (`#315 <https://github.com/dtcenter/METcalcpy/issues/315>`_)
     * Enhanced documentation for Difficulty index (`#332 <https://github.com/dtcenter/METcalcpy/issues/332>`_)

  .. dropdown:: Bugfixes

     * Add missing reliability statistics (`#330 <https://github.com/dtcenter/METcalcpy/issues/330>`_)

METcalcpy Version 3.0.0-beta1 release notes (20230915)
------------------------------------------------------

  .. dropdown:: New Functionality

  .. dropdown:: Enhancements

  .. dropdown:: Internal

  .. dropdown:: Bugfixes

     * Remove reset_index from various calculations (`#322 <https://github.com/dtcenter/METcalcpy/issues/322>`_)


METcalcpy Upgrade Instructions
==============================

Upgrade instructions will be listed here if they are applicable
for this release.
