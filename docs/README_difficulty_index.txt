Difficulty Index
-----------------
Written by Bill Campbell and Liz Satterfield (NRL)
Modified by Lindsay Blank (NCAR)
Date Modified: October 8, 2020

Background
----------
The overall aim of this work is to graphically represent the expected difficulty of a decision based on a set of forecasts (ensemble) of, e.g., significant wave height as a function of space and time. There are two basic factors that can make a decision difficult. The first factor is the proximity of the ensemble mean forecast to a decision threshold, e.g. 12 ft seas. If the ensemble mean is either much lower or much higher than the threshold, the decision is easier; if it is closer to the threshold, the decision is harder. The second factor is the forecast precision, or ensemble spread. The greater the spread around the ensemble mean, the more likely it is that there will be ensemble members both above and below the decision threshold, making the decision harder. (A third factor that we will not address here is undiagnosed systematic error, which adds uncertainty in a similar way to ensemble spread.) The challenge is combing these factors into a continuous function that allows the user to assess relative risk.

Structure of piecewise_linear class:
------------------------------------
Aplin : PiecewiseLinear object (envelope function), essentially a piecewise linear localization function based on muij

Structure of calc_difficulty_index.py:
------------------------------
   1) imports piecewise_linear class (as plin)
   2) Public interface routines are forecast_difficulty()
   3) Main routine is forecast_difficulty(sigmaij,muij,threshold,fieldijn,Aplin,sigma_over_mu_ref,dij)
    	sigmaij : 2D numpy array
       		Positive definite array of standard deviations of a 2D field.
   	muij : Float scalar or 2D numpy array
        	The mean values corresponding to sigmaij.
    	threshold : Float (or int) scalar
        	A significant value to be compared with values of the forecast field
        	for each ensemble member.
    	fieldijn : 3D numpy array
        	Values of the forecast field. Third dimension is ensemble member.
    	Aplin : PiecewiseLinear object (envelope function), optional
        	Essentially a piecewise linear localization function based on muij
    	sigma_over_mu_ref : Scalar, optional
        	Highest value of sigmaij/muij for past 5 days (nominally).

	Returns
    		dij : 2D numpy array
        		Normalized difficulty index ([0,1.5]).
        	Larger (> 0.5) means more difficult
