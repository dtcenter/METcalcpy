import pytest
import numpy as np
import pandas as pd

import metcalcpy.util.mode_3d_volrat_statistics as m3vs

column_names = np.array(["object_type", "volume", "fcst_flag", "simple_flag", "matched_flag"])

data_values = np.array([
        ['3d', 100, 1, 1, 1],
        ['3d', 30, 0, 1, 1],
    ])

funcs = ["volrat_fsa_asa",
"volrat_osa_asa",
"volrat_asm_asa",
"volrat_asu_asa",
"volrat_fsm_fsa",
"volrat_fsu_fsa",
"volrat_osm_osa",
"volrat_osu_osa",
"volrat_fsm_asm",
"volrat_osm_asm",
"volrat_osu_asu",
"volrat_fsa_aaa",
"volrat_osa_aaa",
"volrat_fsa_faa",
"volrat_fca_faa",
"volrat_osa_oaa",
"volrat_oca_oaa",
"volrat_fca_aca",
"volrat_oca_aca",
"volrat_fsa_osa",
"volrat_osa_fsa",
"volrat_aca_asa",
"volrat_asa_aca",
"volrat_fca_fsa",
"volrat_fsa_fca",
"volrat_oca_osa",
"volrat_osa_oca",
"objvhits",
"objvmisses",
"objvfas",
"objvcsi",
"objvpody",
"objvfar"]


def test_calculate_3d_volrat_fsa_asa():
    for f in funcs:
        func_str = f"calculate_3d_{f}"
        func = getattr(m3vs, func_str)
        actual = func(data_values, column_names)
        print(f'"{f}", {actual}')