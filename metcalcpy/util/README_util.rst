Module that contains other utilities that can be used for 'odds and ends'

convert_lon_360_to_180()

=======
  Input:  
  a list or numpy array (floats or ints) of longitudes from the range 0 to 360
  
  Output: 
  a numpy array (float or integer, depending on input) of longitudes from -180 to 180



convert_lons_indices()
   Input:
        lons_in: a list of longitudes to convert

        minlon_in: The minimum value/start value for converted longitudes

        
       
   Returns:
   
      reordered_lons:  sorted array of longitudes
      
      lonsortlocs:  sorted array indices

