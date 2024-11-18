import metcalcpy.agg_stat_bootstrap as asb
from test.utils import AGG_STAT_AND_BOOT_DATA


DEFAULT_CONF = {
    "log_dir": "/tmp",
    "log_filename": "tmp.log",
    "log_level": "DEBUG",
    "agg_stat_input": AGG_STAT_AND_BOOT_DATA,
}


def test_smoke():
    """Basic test to check object instantiation"""
    asb.AggStatBootstrap(DEFAULT_CONF)
