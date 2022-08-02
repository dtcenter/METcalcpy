Contributor's Guide
====================

METcalcpy is written entirely in Python to provide statistics calculations and other utilities that
are used by METviewer, METplotpy, and other applications.

Python Requirements
~~~~~~~~~~~~~~~~~~~

* Python 3.8.6

* cartopy 0.20.2

* cmocean

* eofs

* imutils 0.5.4

* imageio 2.19.2

* lxml

* matplotlib 3.5.1

* metcalcpy

* metpy

* netcdf4 1.5.8

* numpy 1.22.3

* pandas 1.4.2

* pytest 7.1.2

* pyyaml 5.3.1 or above

* scikit-image 0.18.1 or above

* scikit-learn 0.23.2

* scipy 1.8.0

* statsmodels 0.12.2 or above

* xarray 2022.3.0


Retrieve METcalcpy code
~~~~~~~~~~~~~~~~~~~~~~~

You can retrieve the METcalcpy source code using the web browser. Begin by entering
https://github.com/dtcenter/METcalcpy in
the web browser's navigation bar.  On the right-hand side of the web page for the METcalcpy repository, click on 
the `Releases` link.  This leads to a page where all available releases are available.  The latest release will be
located at the top of the page.  Scroll to the release of interest and below it's title is an `Assets` link in small
text.  Click on the inverted triangle to the left of the `Assets` text to access the menu. To download the source code,
click on either the zip or tar.gz version of the source code and save it to a directory where the METcalcpy source code
will reside (e.g. /home/someuser/).

Uncompress the compressed code using unzip <code> for the zip version or tar -xvfz <code> for the tar.gz version.

Coding Standards
~~~~~~~~~~~~~~~~

METplus has adopted some coding standards for its Python code.  Detailed information can be found here: https://metplus.readthedocs.io/en/main_v4.0/Users_Guide/

Comment the Python code using Python docstrings: https://peps.python.org/pep-0257/

Organization of Code in the Github Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The source code for METcalcpy resides in a public GitHub repository:
https://github.com/dtcenter/METcalcpy


Contributed code will reside in one of the following directories:

* METcalcpy/metcalpy

* METcalcpy/metcalcpy/contributed

* METcalcpy/metcalcpy/diagnostics

* METcalcpy/metcalpy/pre-processing

* METcalcpy/metcalcpy/util

The *METcalcpy/metcalcpy/contributed* directory is where contributed code (from outside contributors) is initially saved.

The *METcalcpy/metcalcpy/diagnostics* directory is where any code that is involved with performing diagnostics is to be saved.

The  *METcalcpy/metcalpy/pre-processing* directory is for code that is involved with doing any data pre-processing.

Finally, the *METcalcpy/metcalcpy/util* directory contains code that can be re-used by other Python modules.  In the METcalcpy/metcalcpy directory, there are statistics scripts that are mainly used by METviewer but can be imported by other Python scripts for use.



Tasks to Perform Before Contributing Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Create a Github issue describing what the contribute code will do

* Select  either an **Enhancement request**  or **New feature request**

* Fill out the issue template with relevant information

* Provide information about the enhancement or new feature, the time estimate for the work, assignees (an engineer and a scientist), and fill in the labels (menu on the right hand bar of the issues page).

* Provide as much relevant information as possible

* **NOTE**: The METplus development team and management can assist in filling out some of this information



Contributing Your Code
~~~~~~~~~~~~~~~~~~~~~~

* Clone the METcalcpy repository and checkout the ‘develop’ branch.

* Create a feature branch based on the ‘develop’ branch.

* Employ the naming convention ‘feature_<github feature number>_<brief description>’ such as:

     *feature_123_calculate_xyz*

   for GitHub feature number *123* and that does *xyz*.

* Begin work in this feature branch that you created in the previous step.

* If you are incorporating existing code, copy your code to the *METcalcpy/metcalcpy/contributed* directory.

Otherwise work in one of the appropriate METcalcpy directories.

* Make any necessary changes to your code to conform to the coding conventions

* Migrate it to the code to one of the other, more applicable directories (if you are incorporating pre-existing code).


Testing Your Code
~~~~~~~~~~~~~~~~~~

* Use the pytest framework to create tests to ensure that your code works

 * Refer to *METcalcpy/test* for examples

* Include any sample test data

* If your sample data is large ( >100 MB), contact one of the METcalcpy developers for an alternate (other than Github) storage location

* For sample data <100 MB, save your data in the *METcalcpy/test/data* directory

* Save your test code and any sample data  in the METcalcpy/test directory


Create User Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

* Comment your Python code using python docstrings:

   https://peps.python.org/pep-0257/

* Documentation is located in the *METcalcpy/docs/Users_Guide* and is saved as
  restructured text (.rst)

* You will need to have the following sphinx packages installed on your system or available in your conda/virtualenv:

   * sphinx

   * sphinx-gallery

   * sphinx_rtd_theme


* Verify that your documentation is correct by building it:

  * cd to *METcalcpy/docs/*

* from the command line, run the following commands:
   *build clean*

   *build_html*

* Verify that there aren’t any error messages in the output

* Newly build documentation resides in the *METcalcpy/docs/_build/html/docs* directory

* Visually inspect your documentation with your browser by entering the following in your browser's navigation bar:

   file:///<path/to/METcalcpy_source_code>/feature_123_xyz/METcalcpy/docs/_build/html/Users_Guide/index.html

   where *<path/to/METcalcpy_source_code>* is the directory where you cloned the METcalcpy source code

   and *feature_123_xyz* is the feature branch you created

Incorporate Your Code Into the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Create a pull request (PR) within GitHub and assign one or more scientists and/or engineers from the METplus core team to review your code to verify that your tests are successful and the documentation is correct.

* Update the *METcalcpy/requirements.txt* with any additional Python packages that are needed beyond what is already defined in the requirements.txt file

* Update the *METcalcpy/.github/workflows/unit_tests.yaml* to include any new tests written in pytest to be included in the GitHub actions workflow.

* When your PR has been approved, you (or your reviewer) can merge the code into the *develop* branch

* Close the Github issue you created.




