#export OUTPUT_DIR=/home/minnawin/METplotpy_Data/VertInterp
export OUTPUT_DIR=/d1/personal/minnawin/METplotpy_Output/VertInterp
export DATA_DIR=/home/minnawin/METplotpy_Data/VertInterp
python ../metcalcpy/vertical_interp.py \
    --datadir $DATA_DIR \
    --input tc_rmw_dev_test_out.nc \
    --config height_from_pressure_tcrmw.yaml \
    --output $OUTPUT_DIR/tc_rmw_dev_test_vertical_interp.nc \
#    --debug
