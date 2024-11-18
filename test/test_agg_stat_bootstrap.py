import metcalcpy.agg_stat_bootstrap as asb
from test.utils import AGG_STAT_AND_BOOT_DATA


DEFAULT_CONF = {
    "log_filename": "tmp.log",
    "log_level": "DEBUG",
    "agg_stat_input": AGG_STAT_AND_BOOT_DATA,
}


def test_smoke(tmp_path):
    """Basic test to check object instantiation"""
    DEFAULT_CONF["log_dir"] = tmp_path
    asb.AggStatBootstrap(DEFAULT_CONF)
