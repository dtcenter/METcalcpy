import pytest
from unittest.mock import patch
import numpy as np
import metcalcpy.util.mode_3d_volrat_statistics as m3vs

column_names = np.array(
    ["object_type", "volume", "fcst_flag", "simple_flag", "matched_flag"]
)


data_simple = np.array(
    [
        ["3d", 100, 1, 1, 1],
        ["3d", 30, 0, 1, 1],
    ]
)


data_complex = np.array(
    [
        ["3d", 100, 1, 1, 1],
        ["3d", 30, 0, 1, 1],
        ["3d", 120, 1, 1, 0],
        ["3d", 12, 0, 1, 1],
        ["3d", 1, 1, 0, 1],
        ["3d", 17, 0, 1, 1],
        ["2d", 200, 1, 1, 1],
        ["3d", 66, 0, 1, 0],
    ]
)


@pytest.mark.parametrize(
    "func_name,data_values,expected",
    [
        ("volrat_fsa_asa", data_simple, 0.7692308),
        ("volrat_osa_asa", data_simple, 0.2307692),
        ("volrat_asm_asa", data_simple, 1.0),
        ("volrat_asu_asa", data_simple, None),
        ("volrat_fsm_fsa", data_simple, 1.0),
        ("volrat_fsu_fsa", data_simple, None),
        ("volrat_osm_osa", data_simple, 1.0),
        ("volrat_osu_osa", data_simple, None),
        ("volrat_fsm_asm", data_simple, None),
        ("volrat_osm_asm", data_simple, 0.2307692),
        ("volrat_osu_asu", data_simple, None),
        ("volrat_fsa_aaa", data_simple, 0.7692308),
        ("volrat_osa_aaa", data_simple, 0.2307692),
        ("volrat_fsa_faa", data_simple, 1.0),
        ("volrat_fca_faa", data_simple, None),
        ("volrat_osa_oaa", data_simple, 1.0),
        ("volrat_oca_oaa", data_simple, None),
        ("volrat_fca_aca", data_simple, None),
        ("volrat_oca_aca", data_simple, None),
        ("volrat_fsa_osa", data_simple, 3.3333333),
        ("volrat_osa_fsa", data_simple, 1.0),
        ("volrat_aca_asa", data_simple, None),
        ("volrat_asa_aca", data_simple, None),
        ("volrat_fca_fsa", data_simple, None),
        ("volrat_fsa_fca", data_simple, None),
        ("volrat_oca_osa", data_simple, None),
        ("volrat_osa_oca", data_simple, None),
        ("objvhits", data_simple, 65.0),
        ("objvmisses", data_simple, None),
        ("objvfas", data_simple, None),
        ("objvcsi", data_simple, None),
        ("objvpody", data_simple, None),
        ("objvfar", data_simple, None),
        ("volrat_fsa_asa", data_complex, 0.6376812),
        ("volrat_osa_asa", data_complex, 0.3623188),
        ("volrat_asm_asa", data_complex, 0.4608696),
        ("volrat_asu_asa", data_complex, 0.5391304),
        ("volrat_fsm_fsa", data_complex, 0.4545455),
        ("volrat_fsu_fsa", data_complex, 0.5454545),
        ("volrat_osm_osa", data_complex, 0.472),
        ("volrat_osu_osa", data_complex, 0.528),
        ("volrat_fsm_asm", data_complex, 0.5454545),
        ("volrat_osm_asm", data_complex, 0.3710692),
        ("volrat_osu_asu", data_complex, 0.3548387),
        ("volrat_fsa_aaa", data_complex, 0.6358382),
        ("volrat_osa_aaa", data_complex, 0.3612717),
        ("volrat_fsa_faa", data_complex, 0.9954751),
        ("volrat_fca_faa", data_complex, 0.0045249),
        ("volrat_osa_oaa", data_complex, 1.0),
        ("volrat_oca_oaa", data_complex, None),
        ("volrat_fca_aca", data_complex, 1.0),
        ("volrat_oca_aca", data_complex, None),
        ("volrat_fsa_osa", data_complex, 1.76),
        ("volrat_osa_fsa", data_complex, 1.0),
        ("volrat_aca_asa", data_complex, 0.0028986),
        ("volrat_asa_aca", data_complex, 345.0),
        ("volrat_fca_fsa", data_complex, 0.0045455),
        ("volrat_fsa_fca", data_complex, 220.0),
        ("volrat_oca_osa", data_complex, None),
        ("volrat_osa_oca", data_complex, None),
        ("objvhits", data_complex, 79.5),
        ("objvmisses", data_complex, 66.0),
        ("objvfas", data_complex, 120.0),
        ("objvcsi", data_complex, 0.299435),
        ("objvpody", data_complex, 0.5463918),
        ("objvfar", data_complex, 0.2739726),
    ],
)
def test_calculate_3d_volrat(func_name, data_values, expected):
    func_str = f"calculate_3d_{func_name}"
    func = getattr(m3vs, func_str)
    actual = func(data_values, column_names)
    assert actual == expected

    # Check None is returned on Exception
    with patch.object(m3vs, "rename_column", side_effect=TypeError):
        actual = func(data_values, column_names)
    assert actual == None
