# METcalcpy
Python version of statistics calculation functionality of METviewer, 
METexpress, plotting packages in METplotpy and as stand-alone package for any other application.

Instructions for installing the metcalcpy package locally
---------------------------------------------------------
- activate your conda environment (i.e. 'conda activate your-conda-env-name')
- from within your active conda environment, cd to the METcalcpy/ directory, where you will see the setup.py script
- from this directory, run the following on the command line: pip install -e .
- the -e option stands for editable, which is useful in that you can update your METcalcpy/metcalcpy source without reinstalling it 

- use metcalcpy package via import statement:
  - Examples:
   
    - import metcalcpy.util.ctc_statistics as cstats
        - to use the functions in the ctc_statistics module
  
