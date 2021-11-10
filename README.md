# METcalcpy
Python version of statistics, pre-processing and diagnostics calculation functionality of METviewer, 
METexpress, plotting packages in METplotpy and as stand-alone package for any other application.

Please see the [METcalcpy User's Guide](https://metcalcpy.readthedocs.io/en/latest) for more information.

Support for the METplus components is provided through the
[METplus Discussions](https://github.com/dtcenter/METplus/discussions) forum.
Users are welcome and encouraged to answer or address each other's questions there!  For more
information, please read
"[Welcome to the METplus Components Discussions](https://github.com/dtcenter/METplus/discussions/939)".

Instructions for installing the metcalcpy package locally
---------------------------------------------------------
- activate your conda environment (i.e. 'conda activate your-conda-env-name')
- from within your active conda environment, cd to the METcalcpy/ directory, where you will see the setup.py script
- from this directory, run the following on the command line: pip install -e .
- the -e option stands for editable, which is useful in that you can update your METcalcpy/metcalcpy source without reinstalling it 
- the . indicates that you should search the current directory for the setup.py script

- use metcalcpy package via import statement:
  - Examples:
   
    - import metcalcpy.util.ctc_statistics as cstats
        - to use the functions in the ctc_statistics module
  
