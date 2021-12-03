export DATA_DIR=/path/to/input-data
export OUTPUT_DIR=/path/to/output

python ../metcalcpy/vertical_interp.py \
    --datadir $DATA_DIR \
    --input tc_rmw_example.nc \
    --config height_from_pressure_tcrmw.yaml \
    --output $OUTPUT_DIR/tc_rmw_example_vertical_interp.nc \
    --debug
