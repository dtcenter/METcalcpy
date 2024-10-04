****************
Logging In Guide
****************


This guide provides a comprehensive overview of the newly integrated logging capabilities 
within METcalcpy. These enhancements are designed to provide users with valuable insights 
into the application's execution, aiding in tasks such as debugging, performance monitoring, 
and understanding the operational flow of the program.


What's New
==========

Centralized Logging Configuration (**logging_config.py**):
----------------------------------------------------------

A new script, **logging_config.py**, has been introduced to centralize the management of logging 
configurations. This approach ensures consistency and simplifies the maintenance of logging 
settings across all modules within METcalcpy.


<<<<<<< HEAD
* **Key Feature:** :code:`setup_logging` **Function**
=======
* Key Feature: :code:`setup_logging` function
>>>>>>> develop

  * The :code:`setup_logging` function is the core of **logging_config.py**. It initializes 
    and configures the logger instance based on parameters specified in a YAML configuration 
    file. This function reads logging settings such as :code:`log_dir`, 
    :code:`log_filename`, and :code:`log_level` from the YAML file and sets 
    up Python's logging module accordingly.
  * By isolating the logging configuration in this script, it becomes easier to 
    manage and update logging behavior without altering the core logic of other modules.

Example Integration in **agg_stat.py**:

.. code-block:: py

  from metcalcpy.logging_config import setup_logging
  
  class AggStat:
      def __init__(self, in_params):
          self.logger = setup_logging(in_params)
          # Other initialization code...

In this example, when an :code:`AggStat object` is instantiated, it invokes the 
:code:`setup_logging` function, passing in the :code:`in_params` dictionary, 
which contains logging configurations from a YAML file such as 
**val1l2_agg_stat.yaml**. This ensures the logger is configured according to 
the user's settings.

YAML-Driven Configuration
-------------------------

METcalcpy now allows users to customize logging behavior directly within 
<<<<<<< HEAD
the user's YAML configuration files, eliminating the need for hardcoding 
=======
their YAML configuration files, eliminating the need for hardcoding 
>>>>>>> develop
logging settings in Python scripts.

**Key Parameters in YAML Configuration:**

<<<<<<< HEAD
| :code:`log_dir:` Specifies the directory where log files are stored.
| :code:`log_filename:` Defines the name of the log file.
| :code:`log_level:` Determines the verbosity of the log output. 
  Available levels are DEBUG, INFO, WARNING, and ERROR.
| :code:`log_level:` By setting the appropriate log level in the YAML configuration 
  file (e.g., log_level: WARNING), the user can control the verbosity of the log output, 
  ensuring that only the necessary information is recorded.
=======
:code:`log_dir:` Specifies the directory where log files are stored.

:code:`log_filename:` Defines the name of the log file.

:code:`log_level:` Determines the verbosity of the log output. 
Available levels are :code:`DEBUG, INFO, WARNING, and ERROR:`.

:code:`log_level:` By setting the appropriate log level in your YAML configuration 
file (e.g., log_level: WARNING), you can control the verbosity of the log output, 
ensuring that only the necessary information is recorded.
>>>>>>> develop

METcalcpy supports the following log levels:

  1. **DEBUG:**

    * **Purpose:** Captures detailed information for diagnosing issues.
    * **Use Case:** Ideal during development or troubleshooting to see all 
      the internal workings of the application.

  2. **INFO:**

    * **Purpose:** Records general information about the application's execution.
    * **Use Case:** Suitable for tracking the progress and key events 
      in the application's workflow without overwhelming detail.

  3. **WARNING:**

    * **Purpose:** Logs potential issues that are not immediately critical but 
      could lead to problems.
    * **Use Case:** Useful for highlighting areas that may require attention 
      but don't stop the application from running.

  4. **ERROR:**

    * **Purpose:** Captures serious issues that prevent parts of the 
      application from functioning correctly.
    * **Use Case:** Necessary for logging events that require immediate 
      attention and could cause the application to fail or produce incorrect results.

Informative Log Formatting
--------------------------

Log messages in METcalcpy are meticulously formatted to include detailed information, 
improving readability and facilitating easier analysis of log data.

<<<<<<< HEAD
**Standard Log Format Includes:**

  * **Timestamp (UTC):** Each log message is tagged with a UTC timestamp 
    (e.g., :code:`2023-12-19 18:20:00 UTC`), ensuring consistent timekeeping across systems.
  * **User ID:** The User ID of the script initiator is included, aiding traceability, 
    particularly in multi-user environments.
  * **Log Level:** Indicates the severity of the message 
    (e.g., DEBUG, INFO, WARNING, ERROR).
  * **Log Message:** The main content of the log entry, which may provide context 
    about events or operations within the script.

Safe Logging Utility (safe_log.py)
----------------------------------

A utility function, :code:`safe_log`, is introduced in **safe_log.py** to 
enhance the robustness of logging operations.

  * **Functionality:**

    * The :code:`safe_log` function ensures that logging does not become a point of failure. 
      It checks if a logger object is properly configured before logging any message. If a logger 
      is not available or an error occurs during logging, :code:`safe_log` handles the 
      situation gracefully without interrupting the application's core functionality.

Example Usage in **agg_stat.py**:

.. code-block:: py

  from metcalcpy.util.safe_log import safe_log

  safe_log(self.logger, "info", "Successfully loaded data from ...")

Signal Handling for Graceful Shutdown
-------------------------------------

The **logging_config.py** script is equipped to handle unexpected 
program terminations gracefully by setting up signal handlers.

  * **Supported Signals:**

    * **SIGINT:** Typically triggered by pressing :code:`CTRL+C` to interrupt the program.
    * **SIGTERM:** Sent by other processes to request the program to stop gracefully.

When these signals are intercepted, a message like "Received signal ... Shutting down." 
is logged, providing insight into the cause of the termination. This feature is valuable 
for debugging and system monitoring.

How to Use Logging in METcalcpy
-------------------------------

**Step 1: Configure Logging in the YAML File**

Begin by opening the YAML configuration file (e.g., **val1l2_agg_stat.yaml**) 
and insert the logging parameters at the top level of the YAML file:

| :code:`log_dir: /path/to/your/log/directory`
| :code:`log_filename: my_application_log.txt`
| :code:`log_level: INFO`

**Step 2: Execute METcalcpy Scripts**

With logging configured in the YAML file, run the METcalcpy scripts as usual. 
The logging system will automatically manage log files according to the user's 
specified settings.

**Additional Notes**

  * **UTC Timestamps:** METcalcpy uses UTC for all log timestamps, 
    ensuring consistency across systems and time zones.
  * **Log File Appending:** Logs are appended to existing files when scripts 
    are executed multiple times with the same configuration.

**Example Log Entry:**

:code:`2023-12-19 18:20:00 UTC | user123 | INFO | Data loading completed successfully.`
=======





>>>>>>> develop




