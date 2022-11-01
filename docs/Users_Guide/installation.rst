Installation guide for METcalcpy
===========================================

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

* pandas 1.2.3

* pytest 7.1.2

* pyyaml 5.3.1 or above

* scikit-image 0.18.1 or above

* scikit-learn 0.23.2

* scipy 1.8.0

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

Install METcalcpy package
~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended that one works within a conda environment when using the METcalcpy package.  Please refer to 
https://docs.conda.io/projects/conda/en/latest for more information abount conda as a package and environnent
manager. 

To install METcalcpy, activate your conda environment that contains all the necessary Python packages
listed above in the **Python Requirements** section.  From the command line, cd to the directory where you stored the
METcalcpy source code, e.g. `/User/someuser/METcalcpy`.  From this directory, run the following:

`pip install -e .`

This instructs pip to install the package based on instructios in the setup.py file located in the current directory
(as indicated by the '.').  The `-e` directs pip to install the package in edit mode, so if one wishes to make changes
to this source code, the changes are automatically applied without the need to re-install the package.



`Note: In a future release, METcalcpy will be located on PyPI (Python Package Index) to facilitate
installation into a virtual environment or conda environment using `pip <packagename>`.  `


Explore METcalcpy modules
~~~~~~~~~~~~~~~~~~~~~~~~~

There are numerous statistics tools and other useful utilities available in METcalcpy. To examine what is
available, open a Python console from the command line by entering `python` at the command line.

Verify that this console's version of Python is consistent with the version required in the **Python Requirements**
section by looking at the output::

    % python
    Python 3.6.3 | packaged by conda-forge | (default, Dec  9 2017, 16:20:51)
    [GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>>

The first line of output from the console contains information about Python, followed by information about your
operating system.  In this example, the Python version that was installed in the conda environment is
consistent with the required version.

At the console prompt, enter `import metcalcpy` then enter `return` to import all the packages and modules
in METcalcpy.  If the METcalcpy package was correctly installed in this active conda environment, a new console
prompt is returned::

    >>> import metcalcpy
    >>>


To view all the available modules and packages, enter `help(metcalcpy)`::

  >>> help(metcalcpy)

You should see some output like the following::

    NAME
       metcalcpy - This module contains a variety of statistical calculations.

    PACKAGE CONTENTS
        agg_stat
        agg_stat_bootstrap
        agg_stat_eqz
        agg_stat_event_equalize
        bootstrap_custom
        calc_difficulty_index
        compare_images
        contributed (package)
        event_equalize
        event_equalize_against_values
        piecewise_linear
        sum_stat
        util (package)
        validate_mv_python
        vertical_interp

Packages (which are directories in the source code that contain Python modules) are indicated by `(package)` next to
the name. Enter `q` to return to the console prompt. To find out more about a module of interest, explicitly import it
via `from metcalcpy import <module>` (where <module> is the module of interest).  For example, look at the methods
that are available in the compare_images module::

    >>> from metcalcpy import compare_images
    >>> help(compare_images)

One can access the pydocs (Python documentation) from the compare_images module (compare_images.py) by entering
`help(<module>)`.  This provides valuable information about the module (or package) such as the available methods
and their method signatures (or in the case of packages, any available modules).  Enter `return` or the spacebar
to scroll down to the next line or page of the output.  When finished viewing, enter `q`.

To access other packages, such as the util package from METcalcpy, import it::

    >>> from metcalcpy import util
    >>> help(util)

which give output like this::

    Help on package metcalcpy.util in metcalcpy:

    NAME
       metcalcpy.util

    PACKAGE CONTENTS
        ctc_statistics
        ecnt_statistics
        grad_statistics
        met_stats
        mode_2d_arearat_statistics
        mode_2d_ratio_statistics
        mode_3d_ratio_statistics
        mode_3d_volrat_statistics
        mode_arearat_statistics
        mode_ratio_statistics
        nbrcnt_statistics
        nbrctc_statistics
        pstd_statistics
        rps_statistics
        sal1l2_statistics
        sl1l2_statistics
        ssvar_statistics
        utils
        val1l2_statistics
        vcnt_statistics
        vl1l2_statiatics


To obtain information on the utils module in metcalcpy.util, do the following::

    >>> from metcalcpy.util import utils
    >>> help(utils)

Produces information that looks like the following::

   Help on module metcalcpy.util.utils in metcalcpy.util:

   NAME
       metcalcpy.util.utils - Program Name: met_stats.py

   FUNCTIONS
       aggregate_field_values(series_var_val, input_data_frame, line_type)
         Finds and aggregates statistics for fields with values containing ';'.
         Aggregation  happens by valid and lead times
           These fields are coming from the scorecard and looks like this: vx_mask : ['EAST;NMT'].
           This method finds these values and calculate aggregated stats for them

              Args:
                  series_var_val: dictionary describing the series
                  input_data_frame: Pandas DataFrame
                  line_type: the line type

              Returns:
                  Pandas DataFrame with aggregates statistics

       calc_derived_curve_value(val1, val2, operation)
         Performs the operation with two numpy arrays.
         Operations can be



Using METcalcpy modules
~~~~~~~~~~~~~~~~~~~~~~~

From within the active conda environment, use the METcalcpy packages and
and modules of interest in your code.  For example, in the METplotpy performance_diagram.py file, the event_equalization
method is imported in the following manner::

  import metcalcpy.util.utils as calc_util

which is then used in the code::

    self.input_df = calc_util.perform_event_equalization(self.parameters, self.input_df)











