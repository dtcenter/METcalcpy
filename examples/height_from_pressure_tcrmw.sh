export DATA_DIR=/path/to/data
export INPUT=tc_rmw_example.nc
export OUTPUT=/path/to/output/directory
python ../metcalcpy/vertical_interp.py \
    --datadir $DATA_DIR \
    --input tc_rmw_example.nc \
    --config height_from_pressure_tcrmw.yaml \
    --output $OUTPUT/tc_rmw_dev_test_vertical_interp.nc
