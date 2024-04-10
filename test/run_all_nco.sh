#!/bin/bash

python3 -m pytest test_agg_eclv.py
python3 -m pytest test_agg_ratio.py
python3 -m pytest test_agg_stats_and_boot.py
python3 -m pytest test_agg_stats_with_groups.py
python3 -m pytest test_calc_difficulty_index.py
python3 -m pytest test_convert_lon_indices.py
python3 -m pytest test_ctc_statistics.py
python3 -m pytest test_event_equalize_against_values.py
python3 -m pytest test_event_equalize.py
python3 -m pytest test_grid_diag.py
python3 -m pytest test_lon_360_to_180.py
python3 -m pytest test_scorecard.py
python3 -m pytest test_spacetime.py
python3 -m pytest test_statistics.py
python3 -m pytest test_tost_paired.py
python3 -m pytest test_agg_stat.py
python3 -m pytest test_utils.py
python3 -m pytest test_reformatted_for_agg.py
