****************
Difficulty Index
****************

Description
===========

This module is used to calculate the difficulty of a decision based on a set of forecasts, 
such as an ensemble, for quantities such as wind speed or significant wave height as a 
function of space and time.

Example
=======

An example Use-Case for running Difficulty Index for Wind Speed can be found in the METplus 
documentation.

Decision Difficulty Index Computation
=====================================

Consider the following formulation of a forecast decision difficulty index:

  .. math :: d_{i,j} = \frac{A(\bar{x}_{i,j})}{2}(\frac{(\sigma/\bar{x})_{i,j}}{(\sigma/\bar{x})_{ref}}+[1-\frac{1}{2}|P(x_{i,j}\geq thresh)-P(x_{i,j}<thresh)|])

where :math:`\sigma` is the ensemble standard deviation, :math:`\bar{x}` is the ensemble mean, 
:math:`P(x_{i,j}\geq thresh)` is the ensemble (sample) probability of being greater than or equal 
to the threshold, and  :math:`P(x_{i,j}<thresh)` is the ensemble probability of being less than 
the threshold. The :math:`(\sigma/\bar{x})` expression is a measure of spread normalized by the 
mean, and it allows one to identify situations of truly significant uncertainty. Because the 
difficulty index is defined only for positive definite quantities such as significant wave height, 
division by zero is avoided. :math:`(\sigma/\bar{x})_{ref}` is a (scalar) reference value, for 
example the maximum value of :math:`(\sigma/\bar{x})` obtained over the last 5 days as a function 
of geographic region.

The first term in the outer brackets is large when the uncertainty in the current forecast is 
large relative to a reference. The second term is minimum when all the probability is either 
above or below the threshold, and maximum when the probability is evenly distributed about the 
threshold. So it penalizes the split case, where the ensemble members are close to evenly split on 
either side of the threshold. The A term outside the brackets is a weighting to account for 
heuristic forecast difficulty situations. Its values for winds are given below.

  .. math :: A = 0 if \bar{x} is above 50kt
  .. math :: A = 0 if \bar{x} is below 5kt
  .. math :: A = 1.5 if \bar{x} is between 28kt and 34kt
  .. math :: \text{A} = 1.5 - 1.5[\frac{\bar{x}(kt)-34kt}{16kt}] for 34kt\leq\bar{x}\leq 50kt
  .. math :: \text{A} = 1.5[\frac{\bar{x}(kt)-5kt}{23kt}] for 5kt\leq\bar{x}\leq 28kt

  .. image:: figure/weighting_wave_hgt_difficulty_index.png

The weighting ramps up to a value 1.5 for a value of :math:`x` that is slightly below the threshold. 
This accounts for the notion that a forecast is more difficult when it is slightly below the threshold 
than slightly above. The value of :math:`A` then ramps down to zero for large values of 
:math:`\bar{x}_{i,j}`.


