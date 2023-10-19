****************
Difficulty Index
****************

Description
===========

This module is used to calculate the difficulty of a decision based on a set of forecasts, 
such as an ensemble, for quantities such as wind speed or significant wave height as a 
function ot space and time.

Example
=======

An example Use-Case for running Difficulty Index for Wind Speed can be found in the METplus 
documentation.

Decision Difficulty Index Computation
=====================================

Consider the following formulation of a forecast decision difficulty index:

  .. math :: \text{d_{i,j}} = \frac{A(\bar{x}_{i,j})}{2}(\frac{(sigma/\bar{x})_{i,j}{(sigma/\bar{x})_{ref}}}+[1-\frac{1}{2}|P(x_{i,j}\geq\text{thresh})-P(x_{i,j}<thresh)|]) 
