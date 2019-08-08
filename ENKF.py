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


def explicit_euler(stepnumber, delta, x0, F):


	#print(f'stepnumber: {stepnumber}. delta: {delta}')
	d         = len(x0)
	x_new     = np.zeros(d)
	x_old     = x0 

	for k in range(stepnumber):

		x_new = x_old + delta*Lorenz96(x_old, 0,F)
		x_old = x_new

	return x_new



def integrate(integration_scheme, stepsize, start, stepnumber, t, h, F):

	if integration_scheme == 'euler':

		return explicit_euler(stepnumber, stepsize, start, F)

	elif integration_scheme == 'runge-kutta':

		return runge_kutta(stepnumber, stepsize, start, F)

	else:
		temp  = t*h
		#temp1 = np.arange(temp, temp + stepsize*stepnumber, stepsize)
		temp1 = np.linspace(temp, temp + stepsize*stepnumber, stepnumber)

		return np.transpose(odeint(Lorenz96,start,temp1, args = (F,)))[:,stepnumber-1]


#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------Filter-------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------


# This function is the basic Kalman filter, I saves all process outcomes into an d x K x N matrix.

def my_EnKF(Z, H, start, measure_noise, T, K, d, q, delta, integration_scheme, obs, F):

	process = start
	h       = obs*delta
	ev_process = np.zeros((d,T))
	ev_process[:,0] = 1/K * np.sum(process, axis = 1) 

	for t in range(T-1):

		if np.isnan(ev_process[:,t]).any():
			for remain_t in range(t+1,T):
				ev_process[:,remain_t] = np.nan*np.ones(d)

			return ev_process 
		
		V_hat = forecast(process, d, K, t, delta, integration_scheme, obs, h, F) 	 # prediction for current timestep based on previous timestep
		Z_ens = generate_ensamble(Z[t+1], measure_noise, K, q)						 # artificial ensamble created from the observation
		C_hat = get_C_hat(V_hat, K, d)												 # Covariance matrix 
		process = analyis(V_hat, C_hat, H, Z_ens, K, d, q)							 # progress at time step n is saved in process matrix

		ev_process[:,t+1] = 1/K * np.sum(process, axis = 1) 

	return ev_process


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


def get_C_hat(V_hat, K, d):

	'''
	This function takes the forecast ensemble V_hat, the number of ensemble members K, and the system dimension d.
	It returns the forecast covariance matrix C_hat.
	'''
	V_bar_hat = 1/K * np.sum(V_hat, axis = 1)
	C_hat     = np.zeros((d,d))  

	for k in range(K):

		temp     = V_hat[:,k]-V_bar_hat
		C_hat    = C_hat + np.outer(temp,temp)

	C_hat =  1/(K-1) * C_hat 

	return C_hat 

