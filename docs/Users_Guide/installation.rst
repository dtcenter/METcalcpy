Installation guide for METcalcpy
===========================================

METcalcpy is written entirely in Python to provide statistics calculations and other utilities that
are used by METviewer, METplotpy, and other applications.

Python Requirements
~~~~~~~~~~~~~~~~~~~

The Python requirements for METcalcpy are the same as those for METplotpy

* Python 3.6.3

* bootstrapped 0.0.2 

* cartopy 0.17.0 or above

* imutils 0.5.3

* kaleido 0.0.1

* kiwisolver 1.0.1

* lxml

* matplotlib 3.3.3 or above

* metcalcpy 

* netcdf 1.5.1.2 or above 

* numpy

* pandas

* pillow

* pingouin 0.3.10 or above

* pint
 
* pip

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


Retrieve METcalcpy code
~~~~~~~~~~~~~~~~~~~~~~~

You can retrieve the METcalcpy source code using the web browser. Begin by entering https://github.com/dtcenter/METcalcpy in 
the web browser's navigation bar.  On the right-hand side of the web page for the METcalcpy repository, click on 
the `Releases` link.  This leads to a page where all available releases are available.  The latest release will be located
at the top of the page.  Scroll to the release of interest and below it's title is an `Assets` link in small text.  Click on 
the inverted triangle to the left of the `Assets` text to access the menu. To download the source code, click on either the 
zip or tar.gz version of the source code and save it to a directory where the METcalcpy source code will reside (e.g. /home/someuser/).  

Uncompress the compressed code using unzip <code> for the zip version or tar -xvfz <code> for the tar.gz version.

Install METcalcpy package
~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended that one works within a conda environment when using the METcalcpy package.  Please refer to 
https://docs.conda.io/projects/conda/en/latest for more information abount conda as a package and environnent
manager. 

To install the METcalcpy package, activate your conda environment that contains all the necessary Python packages listed above in 
the **Python Requirements** section.  From the command line, cd to the directory where you stored the METcalcpy source code, e.g.
/User/someuser/METcalcpy.  From this directory, run the following:

`pip install -e .`

This instructs pip to install the package based on instructios in the setup.py file located in the current directory (as indicated by the '.').
The `-e` directs pip to install the package in edit mode, so if one wishes to make changes to this source code, the changes are automatically
applied without the need to re-install the package. 



`Note: In a future release, the METcalcpy package will be located on PyPI (Python Package Index) to facilitate installation of the package
into a virtual environment or conda environment using `pip <packagename>`.  `


Explore METcalcpy modules
~~~~~~~~~~~~~~~~~~~~~~~~~

There are numerous statistics tools and other useful utilities available in the METcalcpy package.


Using METcalcpy modules
~~~~~~~~~~~~~~~~~~~~~~~

From the same active conda environment where the METcalcpy package was installed, work on the code which will be using the various
modules of the METcalcpy package by importing the module(s) of interest.










