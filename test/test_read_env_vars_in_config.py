import os
import re
import yaml.constructor
from metcalcpy.util import read_env_vars_in_config as read_env
import pytest

def test_with_yaml_string():
    """
        Pass in a YAML key:value string
    """

    os.environ['USER'] ='usr'
    os.environ['SRCDIR'] = 'HOME'
    os.environ['DATE'] = "20260729"
    valid_yaml_str = "some_value: !ENV '${USER}/${SRCDIR}/data/${DATE}'"
    valid_expected = os.environ['USER'] + '/' + \
                     os.environ['SRCDIR'] + '/data/' + os.environ['DATE']

    # test a valid entry
    returned = read_env.parse_config(data=valid_yaml_str)

    print(f"expected: {returned}\n")
    assert returned['some_value'] == valid_expected

    # incorrect tag
    incorrect_tag_yaml_str = "some_value: !env '${HOME}'"
    with pytest.raises(yaml.constructor.ConstructorError):
        _ = read_env.parse_config(data=incorrect_tag_yaml_str)

    # missing braces
    missing_brace_yaml_str = "some_value: !env '$HOME'"
    with pytest.raises(yaml.constructor.ConstructorError):
        _ = read_env.parse_config(data=missing_brace_yaml_str)

    # if no env var set, will return the NAME of the env_var
    no_env_var_yaml_str = "some_value: !ENV '${DOESNT_EXIST}'"
    _ = read_env.parse_config(data=no_env_var_yaml_str)
    expected = 'DOESNT_EXIST'
    assert _['some_value'] == expected

    # no env used, return the same yaml str
    plain_yaml_str = "some_value: /home/username_1/data"
    plain_yaml_dict = {'some_value': '/home/username_1/data'}
    _ = read_env.parse_config(data=plain_yaml_str)
    assert _['some_value'] == plain_yaml_dict['some_value']

def test_yaml_file():
    """
        Test parse_config by passing in a yaml config file.  Not necessary
        to test all the same as the above test, simply testing that a specified yaml
        config file works.
    """

    # test with incorrect tag value (!env instead of !ENV)
    cwd = os.getcwd()
    print(f"current working dir: {cwd}")
    yaml_file = os.path.join(cwd, '../test/data/bad_input.yaml')
    with pytest.raises(yaml.constructor.ConstructorError):
        _ = read_env.parse_config(path=yaml_file)

    # non-existent yaml config file specified
    with pytest.raises(FileNotFoundError):
        _ = read_env.parse_config(path='./nonexistent.yaml')

