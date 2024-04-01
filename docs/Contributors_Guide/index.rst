###################
Contributor's Guide
###################

METcalcpy is written entirely in Python to provide statistics calculations and other utilities that
are used by METviewer, METplotpy, and other applications.  The modules and packages can be imported into
other scripts.

Python Requirements
===================

* Python 3.10.4

* imutils 0.5.4

* imageio 2.25.0

* metpy 1.4.0

* netcdf4 1.6.2

* numpy 1.24.1

* opencv-python 4.7.0.68

* pandas 1.4.2

* pytest 7.1.2

* pyyaml 6.0

* scikit-image 0.19.3 

* scipy 1.10.0

* xarray 2023.1.0


Coding Standards
================

METplus has adopted some coding standards for its Python code.  Detailed information can be found here: https://metplus.readthedocs.io/en/main_v4.0/Contributors_Guide/coding_standards.html

Comment the Python code using Python docstrings: https://peps.python.org/pep-0257/

Organization of Code in the Github Repository
=============================================

The source code for METcalcpy resides in a public GitHub repository:
https://github.com/dtcenter/METcalcpy

Contributed code will reside in one of the following directories:

* *METcalcpy/metcalpy*

* *METcalcpy/metcalcpy/contributed*

* *METcalcpy/metcalcpy/diagnostics*

* *METcalcpy/metcalpy/pre-processing*

* *METcalcpy/metcalcpy/util*

The *METcalcpy/metcalcpy/contributed* directory is where contributed code (from outside contributors) is initially saved.

The *METcalcpy/metcalcpy/diagnostics* directory is for code that is involved with performing diagnostics.

The *METcalcpy/metcalpy/pre-processing* directory is for code that is involved with any data pre-processing.

The *METcalcpy/metcalcpy/util* directory contains code that can be re-used by other Python modules.

Finally, the *METcalcpy/metcalcpy* directory contains statistics scripts that are mainly used by METviewer but can be imported by other Python scripts for use.


Tasks to Perform Before Contributing Code
=========================================

* You will need a Github account and be included into the METcalcpy Developer’s group

* Create a Github issue describing what the contribute code will do

* Employ the naming convention ‘feature_<github feature number>_<brief description>’ such as:

     *feature_123_calculate_xyz*

   for GitHub feature number *123* with description *xyz*.

* Select  either an **Enhancement request**  or **New feature request**

* Fill out the issue template with relevant information

* Provide information about the enhancement or new feature, the time estimate for the work, assignees (an engineer and a scientist), and fill in the labels (menu on the right hand bar of the issues page)

* Provide as much relevant information as possible

* **NOTE**: The METplus development team and management can assist in filling out some of this information

* Set up your conda/virtual environment or have your system administrator install the necessary Python version and third-party packages defined above

Retrieve METcalcpy code
=======================

* Create a METcalcpy directory on your host machine (<path-to-METcalcpy>) where you have read/write/execute privileges

  This is where the METcalcpy code will be saved.

   mkdir /home/my_dir/feature_123_xyz

  * In this example, the directory is named after the corresponding Github issue.  This makes it easier to identify which branch is being used.

    *Use a naming convention and directory structure that conforms to your own work flow*

* Change directory to the METcalcpy directory you created::

   cd /home/my_dir/feature_123_xyz/

* Clone the METcalcpy repository from the command line into this directory

    via HTTP::

     git clone https://github.com/dtcenter/METcalcpy

* Change directory to the *METcalcpy* directory::

    cd /home/my_dir/feature_123_xyz/METcalcpy

* The latest major release is the default branch.

   Enter the following at the command line to view the default branch::

     git branch

   You will see something like this:
   main_vn.m

   where *n* and *m* are major and minor version numbers, respectively

* Check out the *develop* branch::

   git checkout develop

* Create a feature branch corresponding to your Github issue::

   git checkout -b feature_123_xyz

   *at this point, the code you have in the feature_123_xyz branch is identical to the code in the develop branch*


Contributing Your Code
======================

* Begin working in the feature branch that you created in the previous step.  From this point on, your code will deviate from the code in the *develop* branch.

* If you are incorporating existing code, copy your code to the *METcalcpy/metcalcpy/contributed* directory.

Otherwise work in one of the appropriate METcalcpy directories.

* Make any necessary changes to your code to conform to the coding conventions

* Migrate it to the code to one of the other, more applicable directories (**if you are incorporating pre-existing code**).


Testing Your Code
=================

* Use the pytest framework to create tests to ensure that your code works

 * Refer to *<path-to-METcalcpy-dir-base>/METcalcpy/test* for examples::

    /home/my_dir/feature_123_xyz/METcalcpy/test

* Include any sample test data

* If your sample data is large ( >100 MB), contact one of the METcalcpy developers for an alternate (other than Github) storage location

* For sample data <100 MB, save your data in the *<path-to-METcalcpy-dir-base>/METcalcpy/test/data* directory::

  /home/my_dir/feature_123_xyz/METcalcpy/test/data


Create User Documentation
=========================

* Comment your Python code using python docstrings:

   https://peps.python.org/pep-0257/

* Documentation is located in the *METcalcpy/docs/Users_Guide* and is saved as
  restructured text (.rst)

* You will need to have the following sphinx packages installed on your system or available in your conda/virtualenv:

   * sphinx

   * sphinx-gallery

   * sphinx_rtd_theme


* Verify that your documentation is correct by building it:

  * cd to */home/my_dir/feature_123_xyz/METcalcpy/docs/*

* from the command line, run the following commands::

   build clean

   build_html

* Verify that there aren’t any warnings or error messages in the output

* Newly build documentation resides in the *METcalcpy/docs/_build/html/docs* directory

* Visually inspect your documentation with your browser by entering the following in your browser's navigation bar:

   file:///<path/to/METcalcpy_source_code>/feature_123_xyz/METcalcpy/docs/_build/html/Users_Guide/index.html

   where *<path/to/METcalcpy_source_code>* is the directory where you cloned the METcalcpy source code

   (e.g. /home/my_dir) and *feature_123_xyz* is the feature branch you created

Incorporate Your Code Into the Repository
=========================================

* Create a pull request (PR) within GitHub and assign one or more scientists and/or engineers from the METplus core team to review your code to verify that your tests are successful and the documentation is correct.

* Update the *METcalcpy/requirements.txt* with any additional Python packages that are needed beyond what is already defined in the requirements.txt file

* Update the *METcalcpy/.github/workflows/unit_tests.yaml* to include any new tests written in pytest to be included in the GitHub actions workflow.

* When your PR has been approved, you (or your reviewer) can merge the code into the *develop* branch

* Close the Github issue you created.




