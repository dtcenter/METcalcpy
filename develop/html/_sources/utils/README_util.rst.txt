util module

convert_lon_360_to_180(longitude)

    Description:  Takes an array (numpy or integer or float array) ranging from 0 to 360 degrees and converts these to -180 to 180 degrees

convert_lons_indices(lons_in, minlon_in, range_in)

    Description: Takes a numpy array as input, and reorders them based on a minimum lon value and the number of longitude values

    Returns:  A tuple, the reordered longitudes and an array of the indices of the original array of longitudes

