import logging
import os
import sys
import getpass
import signal
import time

class UserIDFormatter(logging.Formatter):
    """
    Custom formatter to add user_id in place of the logger name.
    """
    def __init__(self, user_id, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.user_id = user_id

    def format(self, record):
        # Override the 'name' attribute with user_id
        record.name = self.user_id
        return super().format(record)

def handle_signals(signum, frame):
    """
    Handle signals to perform clean shutdown or other custom actions.
    """
    logger = logging.getLogger()
    logger.warning(f'Received signal {signal.strsignal(signum)}. Shutting down.')
    sys.exit(0)

def setup_logging(config_params):
    """
    Set up logging based on the configuration from a YAML file.
    
    Args:
        config_params (dict): The dictionary containing logging configuration (log directory, filename, level).
        
    Returns:
        logger (logging.Logger): Configured logger.
    """
    # Get user ID and command line
    user_id = getpass.getuser()
    command_line = " ".join(sys.argv)
    # Create log directory if it doesn't exist, using the path from the config
    log_dir = config_params.get('log_dir')  # No default here, expect it from YAML
    if not log_dir:
        log_dir = './logs'  # Set default only if not provided
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set log filename, incorporating the log directory path from the config
    log_filename = config_params.get('log_filename')  # No default here, expect it from YAML
    if not log_filename:
        log_filename = 'application.log'  # Set default only if not provided
    log_file = os.path.join(log_dir, log_filename)

    # Set log level from YAML or use default; convert to appropriate logging level
    log_level = config_params.get('log_level')  # No default here, expect it from YAML
    if not log_level:
        log_level = 'WARNING'  # Set default only if not provided
    log_level = log_level.upper()


    # Create a custom formatter that uses UTC for date and includes user_id instead of logger name
    # Add ' UTC' to the format string for the time
    formatter = UserIDFormatter(
        user_id=user_id,
        fmt='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set up logging to write to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.addHandler(file_handler)

    # Set logger to use UTC time
    logging.Formatter.converter = time.gmtime

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    logger.info(f"User: {user_id} has started the script with command: {command_line}")

    return logger

