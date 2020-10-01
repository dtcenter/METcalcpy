python ../metcalcpy/vertical_interp.py \
    --datadir $DATA_DIR/Events/2019-03-13-GreatPlainsCyclone \
    --input MERRA2_400.inst3_3d_asm_Np.20190311.nc4 \
    --config vertical_interp_merra2.yaml \
    --output MERRA2_400.inst3_3d_asm_interp.20190311.nc4 \
    --debug
