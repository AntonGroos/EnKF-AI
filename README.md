# EnKF-AI
This repository contains Python 3 implementaitions of ensemble Kalman Filters with adaptive covariance inflation. Note that the programms in this repository were made to highlight catastrophic filter divergence, thus they are not very efficient and sometimes diliberatly inefficient in choice of some variables.

In order to run the programs contained in this repository you need at least Python 3.7 and following modules:

- numpy 
- tkinter
- matplotlib
- scipy
- pillow
- csv
- threading

There should be eight Python files, three .jpg files, and one .txt file. I will now list the Python files and their purpose:

equilibrium.py given a turbulence regime F the programm starts N = 100 long running simulations of the Lorenz 96 model to and calculates the equilibrium mean, the variance of each node, and the benchmark Error_A. 

experiments.py is a tiny programm, that will help you create a matrix of measurement errors in a shape that the algorithm in plotfilters.py will accept. You don't need to do this unless You want so experiment with higher system dimensions.

ENKF.py the basic ensemble Kalman filter. 
EnKF_CI.py the ensemble Kalman filter with constnant additive covariance inflation.
EnKF_AI.py the ensemble Kalman filter with adaptive covariance inflation.
EnKF_CAI.py the ensemble Kalman Filter with constant additve and adaptive covariance inflation.
These programms won't run on their own and contain only functions and numeric integration schemes.

tests.py this file contains the GUI. Executing it will open a window letting the user enter various variables and start long time simulations or individual trials displaying a plot and statistics. It uses the ENKF.py, EnKF_CI.py, EnKF_AI.py, and EnKF_CAI.py programms for the filter and should be very user friendly.

plotfilters.py uses the Error_matrix.txt to create long time simmulations of each filter, it uses a slightly modified version of the other filter programms (as a function in plotfilters.py) to allow the same measurement errors for all filter for a more accurate comparison. It is also the programm I used to create plots with two filters at the same time for my thesis (if you just want to see the plot set N=1 in the programm). 

More detailed explanations can be found in each of the programms.





