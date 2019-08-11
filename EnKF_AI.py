#!/usr/bin/python3.7
import sys
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def Lorenz96(x0, t, F):
  
	d =  len(x0)
	x = np.zeros(d)

	x[0] = (x0[1] - x0[d-2]) * x0[d-1] -x0[0] + F 
	x[1] = (x0[2] - x0[d-1]) * x0[0] - x0[1] + F
	x[d-1] = (x0[0] - x0[d-3]) * x0[d-2] - x0[d-1] + F

	for j in range(2,d-1):

		x[j] = (x0[j+1] - x0[j-2]) * x0[j-1] - x0[j] + F

	return x

# The RK4 numeric integration scheme
def runge_kutta(stepnumber, delta, x0, F):

	d 		= len(x0)
	x_new 	= np.zeros(d)
	x_old   = x0

	for j in range(stepnumber):

		k1 = delta * Lorenz96(x_old, 0, F)
		k2 = delta * Lorenz96(x_old + k1/2, 0, F)
		k3 = delta * Lorenz96(x_old + k2/2, 0, F)
		k4 = delta * Lorenz96(x_old + k3, 0, F)

		x_new = x_old + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

		x_old = x_new

	return x_new

# The explicit Euler method for numeric integration
def explicit_euler(stepnumber, delta, x0, F):

	d         = len(x0)
	x_new     = np.zeros(d)
	x_old     = x0

	for k in range(stepnumber):

		x_new = x_old + delta*Lorenz96(x_old, 0,F)

		x_old = x_new

	return x_new



def integrate(integration_scheme, stepsize, start, stepnumber, t, h, F):

	'''
	This function formats it's input into a shape, that will be accepted from the choosen
	integration scheme.
	'''

	if integration_scheme == 'euler':

		return explicit_euler(stepnumber, stepsize, start, F)
		

	elif integration_scheme == 'runge-kutta':

		return runge_kutta(stepnumber, stepsize, start, F)

	else:

		temp  = t*h
		temp1 = np.linspace(temp, temp + stepsize*stepnumber, stepnumber)

		return np.transpose(odeint(Lorenz96,start,temp1, args = (F,)))[:,stepnumber-1]


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------Filter-------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------



def my_EnKF(Z, H, start, measure_noise, T, K, d, q, delta, integration_scheme, obs, F):

	'''
	This function is the base of the ensemble kalman filter, it in each iteration it calls the functions
	forecast and analysis to simulate to filter. During this function we also collect all sorts of usefull
	statistics.
	'''

	M1, M2  = get_thresholds(F, q)
	process = start
	h       = obs*delta
	ev_process = np.zeros((d,T))
	ev_process[:,0] = 1/K * np.sum(process, axis = 1) 
	counter    = 0
	average_Theta = 0
	average_Xi = 0

	for t in range(T-1):

		# Here we check, if catastrophic divergence occoured in the last step and if so stops calcultaions for this filter.
		if np.isnan(ev_process[:,t]).any():
			for remain_t in range(t+1,T):

				ev_process[:,remain_t] = np.nan*np.ones(d)

			return ev_process, counter, average_Theta, average_Xi

		
		V_hat = forecast(process, d, K, t, delta, integration_scheme, obs, h, F) 			
		Z_ens = generate_ensamble(Z[t+1], measure_noise, K, q)						

		Theta, Xi  = get_greek(V_hat, Z_ens, H, d, K, q)
		lambda_adap = 0 

		# We only want to save the data for Xi and Theta after the filter has stabilized. The sqrt() is because in this code we actually compare 
		# Theta^2 with M_1^2.
		if t > T//2:
			average_Theta +=  np.sqrt(Theta)
			average_Xi    += Xi 

		# Triggers adaptive inflation, if necessary
		if Theta > M1 or Xi > M2:

			counter += 1
			lambda_adap = Theta*(1+Xi)


		C_hat = get_C_hat(V_hat, K, d, lambda_adap)							
		process = analyis(V_hat, C_hat, H, Z_ens, K, d, q)				
		ev_process[:,t+1] = 1/K * np.sum(process, axis = 1) 


	average_Xi    = average_Xi*2/T
	average_Theta = average_Theta*2/T

	return ev_process, counter, average_Theta, average_Xi


def forecast(V_0, d, K, t, delta, integration_scheme, obs, h, F):

	'''
	This function takes the posterior ensemble from the last step V_0, the system dimension d, the number of ensemble members K,
	the current timestep t, the stepsize of the numeric integrator delta, the stepnumber of the numeric integration obs,
	the numeric integration scheme intergation_scheme, the time between observations , and the turbulence regieme F.

	It uses the integrate function the create the forecast ensemble V_hat = forecast.
	'''

	forecast = np.zeros((d,K))

	for k in range(K):

		forecast[:,k] = integrate(integration_scheme, delta, V_0[:,k], obs, t, h, F)


	return forecast


def generate_ensamble(Z, measure_noise, K, q):

	'''
	This function takes the observation Z, the measurement noise expected value and variance measure_noise, 
	the number of ensemble memebers K, and the number of observed components q.
	It returns the pertubations of the observation Z_new = z_ens.
	'''

	Z_new = np.zeros((q,K))

	for k in range(K):

		Z_new[:,k] =  Z + np.random.normal(measure_noise[0], measure_noise[1], q) 

	return Z_new


def analyis(V_hat, C, H, Z_ens, K, d, q):

	'''
	This function takes the forecast ensemble V_hat, the forecast covariance matrix C, the observation matrix H,
	the ensemble of the observation Z_ens, the number of ensemble members K, the system dimension d, and the
	number of observed components q.
	It returns the posterior ensemble V, saved into process. 
	'''

	V     = np.zeros((d,K))

	for k in range(K):

		V[:,k] =  V_hat[:,k] - dot(dot(dot(C,np.transpose(H)), inv(np.eye(q) + dot(H, dot(C,np.transpose(H))))), dot(H,V_hat[:,k])- Z_ens[:,k])

	return V


def get_C_hat(V_hat, K, d, lambda_adap):

	'''
	This function takes the forecast ensemble V_hat, the number of ensemble members K, the system dimension d,
	and the strenght of the adaptive infaltion lambda_adap.
	It returns the forecast covariance matrix C_hat with the added adaptive inflation.
	'''

	V_bar_hat = 1/K * np.sum(V_hat, axis = 1)
	C_hat     = np.zeros((d,d))  

	for k in range(K):

		temp     = V_hat[:,k]-V_bar_hat
		C_hat    += np.outer(temp,temp)

	C_hat =  1/(K-1) * C_hat + lambda_adap*np.eye(d)

	return C_hat 


def get_thresholds(F, q):

	'''
	This function takes the turbulenz regime F and the number of observed components q
	and returns the thresholds M1 and M2 that where calculated previously in equilibrium.py.
	Following values have been caluclated by me:

	F = 4, 8, 16
	q = 1,2,3,4,5
	d = 5

	any other values will go back to a default value, results may vary.
	'''

	if F == 4:
		if q == 1:

			M1 = 22.45
			M2 = 7.47

		elif q == 2:

			M1 = 17.02
			M2 = 4.21

		elif q == 3:

			M1 = 14.75
			M2 = 2.85

		elif q == 4:

			M1 = 13.95
			M2 = 2.37

		elif q == 5:

			M1 = 12.81
			M2 = 1.68

		else: 

			M1 = 22.45
			M2 = 7.47

	elif F == 8:

		if q == 1:

			M1 = 60.36
			M2 = 30.21

		elif q == 2:

			M1 = 41.83
			M2 = 19.10

		elif q == 3:

			M1 = 33.55
			M2 = 14.1

		elif q == 4:

			M1 = 14.48
			M2 = 2.69

		elif q == 5:

			M1 = 22.45
			M2 = 7.47

		else: 

			M1 = 60.36
			M2 = 30.21


	elif F == 16:

		if q == 1:

			M1 = 153.55
			M2 = 86.13

		elif q == 2:

			M1 = 123.21
			M2 = 67.69

		elif q == 3:

			M1 = 82.08
			M2 = 43.30

		elif q == 4:

			M1 = 46.65
			M2 = 21.30

		elif q == 5:

			M1 = 14.82
			M2 = 2.92

		else: 

			M1 = 153.55
			M2 = 86.13

	else:	

		M1 = 40
		M2 = 15

	return M1, M2


def get_greek(V_hat, Z_ens, H, d, K, q):

	'''
	This function takes the forecast ensemble V_hat, the observation ensemble Z_ens, observation matrix H,
	system dimension d, number of ensemble members K, and number of observed components q.
	It returns the statistics Theta and Xi, that are used to decide whether to trigger adaptive infaltion.   
	'''

	Theta = 0

	for k in range(K):

		Theta += np.linalg.norm(dot(H,V_hat[:,k])-Z_ens[:,k], ord = 2)**2

	Theta = 1/K*Theta


	X_hat 	  = np.array([V_hat[j,:] for j in range(q)])
	X_bar_hat = 1/K * np.sum(X_hat, axis = 1)

	if d > q:

		Y_hat	  = np.array([V_hat[j,:] for j in range(q,d)])
		Y_bar_hat = 1/K * np.sum(Y_hat, axis =1)

		B_hat = np.zeros((q,d-q))

		for k in range(K):

			B_hat += np.outer(X_hat[:,k]-X_bar_hat, Y_hat[:,k]-Y_bar_hat)

		B_hat = 1/(K-1)* B_hat

		Xi = np.linalg.norm(B_hat, ord =2)

	else:

		Xi = 0

	return(Theta, Xi)






