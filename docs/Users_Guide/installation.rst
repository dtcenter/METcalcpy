Installation guide for METplotpy
===========================================

METplotpy is written entirely in Python and uses YAML configuration files and relies
on the METcalcpy package.

Python Requirements
~~~~~~~~~~~~~~~~~~~

* Python 3.6.3

* cartopy 0.17.0 or above

* kaleido

* kiwisolver 1.0.1

* lxml

* matplotlib 3.3 or above

* metcalcpy 

* netcdf 

* numpy

* pandas

* Pillow

* pingouin

* Pint

* pyshp

* plotly 4.9.0

* psutil

* pymysql

* pyshp

* pytest 5.2.1

* PyYAML

* requests

* retrying

* scikit-image 0.16.2

* scipy


Install METcalcpy in your conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended method for installation.

Clone the METcalcpy repository from https://github.com/dtcenter/METcalcpy

From within your *active* conda environment, cd to the METcalcpy/ directory.  This is the directory
where you cloned the METcalcpy repository. In this directory, you should see a setup.py script

From the command line, run *pip install -e .*

Do NOT forget the ending **'.'**  this indicates that you should use the setup.py in the current working directory.
 
The *-e* option allows this installation to be editable, which is useful if you plan on updating your METcalcpy/metcalcpy
source code.  This allows you to avoid reinstalling if you make any changes to your METcalcpy code.

Setting up your PYTHONPATH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a workaround for users who can not or do not have permission to create conda environments.

$METCALCPY_SOURCE is the path to where you downloaded/cloned the METcalcpy code.

**command for csh:** 

setenv PYTHONPATH $METCALCPY_SOURCE/METcalcpy:$METCALCPY_SOURCE/METcalcpy/util:${PYTHONPATH}

**command for bash:**

export PYTHONPATH=\

$METCALCPY_SOURCE/METcalcpy:$METCALCPY_SOURCE/METcalcpy/util:${PYTHONPATH}













