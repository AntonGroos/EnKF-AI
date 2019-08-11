#!/usr/bin/python3.7
import sys
import numpy as np
import tkinter as tk
import csv
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''										!!DISCLAIMER!!
This file was created for the specific purpose of creating the plots and data for my thesis
and is not stable upon change of variables. This is due to the usage a seperate matrix containing 
the values of the observation error. Due to its specific size, changes  may break the 
programm unless the matrix in the file is adjusted as well. Values that can be changed at will are:

F the forcing paramete
q the number of observed components
M1, M2 as well as sigma, mu 
rho the strenght of the constant inflation found in my_EnKF()
scheme the integration scheme

Do not change h or T for reasons above. If You want to do test concerning those, go to tests.py.

For N = 100 tials this took approximatly 6 hours on my system, so plan accordingly.

'''



#---------------------------------------------------------------------------------
#-------------------------------xi - Error values --------------------------------
#---------------------------------------------------------------------------------

csv.field_size_limit(sys.maxsize)
f = open('Error_matrix.txt', 'r', newline = '')

read_file = csv.reader(f, delimiter = ',')

Error_matrix = np.array(list(read_file)).astype("float")




#---------------------------------------------------------------------------------
#---------------------------------------Filter-------------------------------------
#---------------------------------------------------------------------------------

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


def my_EnKF(Z, H, start, measure_noise, T, K, d, q, delta, integration_scheme, obs, F, adap_inf = 0, const_inf = 0):

	'''
	This function is the base of the ensemble kalman filter, it in each iteration it calls the functions
	forecast and analysis to simulate to filter. During this function we also collect all sorts of usefull
	statistics.
	'''

	M1, M2 = get_thresholds(F, q)
	process = start
	h       = obs*delta
	ev_process = np.zeros((d, T))
	ev_process[:, 0] = 1/K*np.sum(process, axis = 1) 
	counter = 0 
	average_Theta = 0
	average_Xi = 0
	rho = 0

	if const_inf:

		rho = 0.1
	
	for t in range(T-1):

		if np.isnan(ev_process[:, t]).any():
			for remain_t in range(t+1, T):
				ev_process[:,remain_t] = np.nan*np.ones(d)

			return ev_process, counter, average_Theta, average_Xi 

		
		V_hat = forecast(process, d, K, t, delta, integration_scheme, obs, h, F) 			# prediction for current timestep based on previous timestep
		Z_ens = generate_ensamble(Z[t+1], measure_noise, K, q, t)								# artificial ensamble created from the observation

		Theta, Xi = 0, 0

		if adap_inf:

			Theta, Xi  = get_greek(V_hat, Z_ens, H, d, K, q)

		lambda_adap = 0 

		# We only want to save the data for Xi and Theta after the filter has stabilized. The sqrt() is because in this code we actually compare 
		# Theta^2 with M_1^2.
		if t > T//2:

			average_Theta +=  np.sqrt(Theta)
			average_Xi    += Xi 


		if (Theta > M1 or Xi > M2) and adap_inf:


			counter += 1

			lambda_adap = Theta*(1+Xi)

		C_hat = get_C_hat(V_hat, K, d, lambda_adap, rho)			# Covariance matrix 
		process = analyis(V_hat, C_hat, H, Z_ens, K, d, q)			# progress at time step n is saved in process matrix
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


def generate_ensamble(Z, measure_noise, K, q, timestamp):

	'''
	This function takes the observation Z, the measurement noise expected value and variance measure_noise, 
	the number of ensemble memebers K, and the number of observed components q.
	It returns the pertubations of the observation Z_new = z_ens.
	'''

	Z_new = np.zeros((q,K))

	for k in range(K):

		Z_new[:,k] =  Z +  measure_noise[timestamp*(K + 1) + (k+1): timestamp*(K + 1) + (k + 1) + q]

	return Z_new


def analyis(V_hat, C, H, Z, K, d, q):

	'''
	This function takes the forecast ensemble V_hat, the forecast covariance matrix C, the observation matrix H,
	the ensemble of the observation Z_ens, the number of ensemble members K, the system dimension d, and the
	number of observed components q.
	It returns the posterior ensemble V, saved into process. 
	'''

	V     = np.zeros((d,K))

	for k in range(K):

		V[:,k] =  V_hat[:,k] - dot(dot(dot(C,np.transpose(H)), inv(0.01*np.eye(q)+ dot(H, dot(C,np.transpose(H))))), dot(H,V_hat[:,k])- Z[:,k])

	return V


def get_C_hat(V_hat, K, d, lambda_adap, rho):

	'''
	This function takes the forecast ensemble V_hat, the number of ensemble members K, the system dimension d,
	and the strenght of the adaptive infaltion lambda_adap.
	It returns the forecast covariance matrix C_hat with the added adaptive and constant inflation.
	'''

	V_bar_hat = 1/K * np.sum(V_hat, axis = 1)
	C_hat     = np.zeros((d,d))  

	for k in range(K):

		temp     = V_hat[:,k]-V_bar_hat
		C_hat    += np.outer(temp,temp)

	C_hat =  1/(K-1) * C_hat + (lambda_adap + rho)*np.eye(d)  

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

	Theta = 0

	for k in range(K):

		Theta += np.linalg.norm(dot(H,V_hat[:,k])-Z_ens[:,k])**2

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

def evaluate_root_mean_square_error(U, V):

	"""
	This function calculates the root mean square error using:
	U the truth
	V the mean of the ensemble 
	"""

	d, S  = np.shape(U)
	temp  = S//2
	temp1 = np.zeros((d, temp))

	for n in range(temp,S): 
		for j in range(d):
			temp1[j,n-temp] = (U[j,n] - V[j,n])**2

	error = np.sqrt(2/(S) * np.sum(temp1, axis = (0,1)))

	return error

def evaluate_pattern_correlation(U, V, T, U_bar, h):

	"""
	This function calculates the pattern correlation using:
	U the truth
	V the mean of the ensemble
	T the time length
	U_bar a vector with the same dimension as U containing the equilibrium mean of the system
	h the time between observations
	"""

	d, S = np.shape(U)
	temp  = T//2
	temp1 = np.zeros(temp)

	for n in range(temp,T):
		m = int(n/h)
		temp1[n-temp] = (np.dot(V[:,m]-U_bar, U[:,m]- U_bar))/((np.linalg.norm(V[:,m]-U_bar, ord = 2) * np.linalg.norm(U[:,m]- U_bar, ord = 2)))

	pattern_cor = 2/T * np.sum(temp1)

	return pattern_cor

def evaluate_posterior_error (U,V,Time):

	"""
	This function calculates the posterior error used mainly for plotting with:
	U the truth
	V the mean of the ensemble
	Time the number of plot points
	"""

	d, T  = np.shape(U)
	error = np.zeros(Time)

	for n in range(Time):

		error[n] = np.linalg.norm((U[:,n]-V[:,n]), ord = 2)

	return error



#---------------------------------------------------------------------------------
#--------------------------------Variables----------------------------------------
#---------------------------------------------------------------------------------

#+++++++++++++++ F value dependent estimates of equilibrium measure+++++++++++++++++
'''
An important note is, that You may have to adjust the value of sigma, if You observe more components. 
The more components you observe the smaller the variance. A solution to this is to choose the first observation
with a very small error for all components we observe. This is currently implemeted and can be found in init_values.
'''
F = 16                  # Forcing, i.e strength of turbulence
d = 5                  # Number of particles


if F == 4:
	mu, sigma      = 1.22, 3.373  
	Benchmark_RMSE = 3.53	 
	ymax           = 4              

elif F == 8:
	mu, sigma      = 2.27, 13.464 
	Benchmark_RMSE = 7.10
	ymax           = 25

elif F == 16:
	mu, sigma      = 3.2, 39.04
	Benchmark_RMSE = 11.98
	ymax 		   = 80

else:
	mu, sigma      = 2.0, 1
	Benchmark_RMSE = 5.0
	ymax 		   = 20
	print('please select a F value out of 4,8,16')

#+++++++++++++++++++++++++++++The Lorenz '96 model++++++++++++++++++++++++++++++++

N = 10
x0 = np.random.normal(mu, sigma, d)	  


#++++++++++++++++++++++++++The Intergation Scheme+++++++++++++++++++++++++++++++++

delta  = 0.0001		   # How fine the numerical integration schould be
scheme = 'euler' 	   # the integration scheme, choose between 'runge-kutta' and 'euler'

#+++++++++++++++++++++++++The Ensamble Kalman Filter++++++++++++++++++++++++++++++

T = 100               # Runtime choose an int
h = 0.05			  # Time between observations of the signal
obs  = int(h/delta)	  # Numerical interation steps inbetween observations  !Do not change!
Time = int(T/h)		  # Number of observation points  !Do not change!

signal = np.transpose(odeint(Lorenz96, x0, np.linspace(0,T,int(T/delta)), args = (F,))) # The signal !Do not change!
U = signal[:,0::obs]
					                   		# Observation times of the signal !Do not change!

K = 6				  # Memebers of the ensamble
q = 1				  # observed components

if d == q:
	H = np.eye(q)
else:
	H = np.concatenate((np.eye(q), np.zeros((q,d-q))), axis = 1)	# Observation matrix 



# ------------------------------ Now we start the N = 100 simulations and collect the according data for the 4 Filters-------------------------


# These are all the statistics we want to collect over the comming trials.

EnKF_divergencies = 0
EnKF_AI_divergencies = 0
EnKF_CI_divergencies = 0
EnKF_CAI_divergencies = 0

EnKF_RMSE = 0
EnKF_AI_RMSE = 0
EnKF_CI_RMSE = 0
EnKF_CAI_RMSE = 0

EnKF_Cor = 0
EnKF_AI_Cor = 0
EnKF_CI_Cor = 0
EnKF_CAI_Cor = 0

EnKF_AI_total_avrg_trigger = 0
EnFK_CAI_total_avrg_trigger = 0

EnKF_AI_triggered_trials = 0
EnKF_CAI_triggered_trials = 0

AI_average_Theta = 0
CAI_average_Theta = 0

AI_average_Xi = 0
CAI_average_Xi = 0


for n in range(N):

	# Since this takes a long time, there is a print command to tell you how far we progressed.
	print(f'Step {n+1} out of {N}.')

	observation = np.array([dot(H, U[:, t]) + Error_matrix[t*(K + 1):t*(K + 1) + q, n] for t in range(Time)])

	init_values = np.zeros((d, K)) + np.random.normal(mu, sigma, (d, K)) #Starting values of the Filter members

	# Here we set the values of the starting ensemble of all observed components to the observation with a small error.
	init_values[0:q,:] = np.transpose(np.array([observation[0] for k in range(K)])) + np.random.normal(0, 0.01, (q, K))


	EnKF_values, not_available_0, not_available_1, not_available_2 = my_EnKF(observation, H, init_values, Error_matrix[:,n], Time, K, d, q, delta, scheme, obs, F)

	EnKF_AI_values, EnKF_AI_triggers, AI_Theta, AI_Xi  = my_EnKF(observation, H, init_values, Error_matrix[:,n], Time, K, d, q, delta, scheme, obs, F, adap_inf = 1)

	EnKF_CI_values, not_available_0, not_available_1, not_available_2 = my_EnKF(observation, H, init_values, Error_matrix[:,n], Time, K, d, q, delta, scheme, obs, F, const_inf = 1)

	EnKF_CAI_values, EnKF_CAI_triggers, CAI_Theta, CAI_Xi = my_EnKF(observation, H, init_values, Error_matrix[:,n], Time, K, d, q, delta, scheme, obs, F, adap_inf = 1, const_inf = 1)

	RMSE_EnKF = evaluate_root_mean_square_error(U, EnKF_values)
	RMSE_EnKF_AI = evaluate_root_mean_square_error(U, EnKF_AI_values)
	RMSE_EnKF_CI = evaluate_root_mean_square_error(U, EnKF_CI_values)
	RMSE_EnKF_CAI = evaluate_root_mean_square_error(U, EnKF_CAI_values)

	if np.isnan(RMSE_EnKF):

		EnKF_divergencies += 1

	if np.isnan(RMSE_EnKF_AI):

		EnKF_AI_divergencies += 1

	if np.isnan(RMSE_EnKF_CI):

		EnKF_CI_divergencies += 1

	if np.isnan(RMSE_EnKF_CAI):

		EnKF_CAI_divergencies +=1


	EnKF_RMSE += RMSE_EnKF*1/N
	EnKF_AI_RMSE += RMSE_EnKF_AI*1/N
	EnKF_CI_RMSE += RMSE_EnKF_CI*1/N
	EnKF_CAI_RMSE += RMSE_EnKF_CAI*1/N

	Cor_EnKF = evaluate_pattern_correlation(U, EnKF_values, T, mu*np.ones(d), h)
	Cor_EnKF_AI = evaluate_pattern_correlation(U, EnKF_AI_values, T, mu*np.ones(d), h)
	Cor_EnKF_CI = evaluate_pattern_correlation(U, EnKF_CI_values, T, mu*np.ones(d), h)
	Cor_EnKF_CAI = evaluate_pattern_correlation(U, EnKF_CAI_values, T, mu*np.ones(d), h)

	EnKF_Cor += Cor_EnKF*1/N
	EnKF_AI_Cor += Cor_EnKF_AI*1/N
	EnKF_CI_Cor += Cor_EnKF_CI*1/N
	EnKF_CAI_Cor += Cor_EnKF_CAI*1/N

	EnKF_AI_total_avrg_trigger += EnKF_AI_triggers*1/N
	EnFK_CAI_total_avrg_trigger += EnKF_CAI_triggers*1/N

	if EnKF_AI_triggers > 0:

		EnKF_AI_triggered_trials += 1

	if EnKF_CAI_triggers > 0:

		EnKF_CAI_triggered_trials += 1

	AI_average_Theta += AI_Theta*1/N
	AI_average_Xi += AI_Xi*1/N

	CAI_average_Theta += CAI_Theta*1/N
	CAI_average_Xi += CAI_Xi*1/N

print(f'Filter divergencies: \nEnKF: {EnKF_divergencies}\nEnKF_AI: {EnKF_AI_divergencies}\nEnKF_CI: {EnKF_CI_divergencies}\nEnKF_CAI: {EnKF_CAI_divergencies}')
print(f'Average RMSE: \nEnKF: {EnKF_RMSE}\nEnKF_AI: {EnKF_AI_RMSE}\nEnKF_CI: {EnKF_CI_RMSE}\nEnKF_CAI: {EnKF_CAI_RMSE}')
print(f'Average pattern correlation: \nEnKF: {EnKF_Cor}\nEnKF_AI: {EnKF_AI_Cor}\nEnKF_CI: {EnKF_CI_Cor}\nEnKF_CAI: {EnKF_CAI_Cor}')
print(f'Adap. inflation triggers on average: \nEnKF_AI: {EnKF_AI_total_avrg_trigger}\nEnKF_CAI: {EnFK_CAI_total_avrg_trigger}')
print(f'Trials with triggered adap. inflation: \nEnKF_AI: {EnKF_AI_triggered_trials}\nEnKF_CAI: {EnKF_CAI_triggered_trials}')
print(f'Average Theta: \nEnKF_AI: {AI_average_Theta}\nEnKF_CAI: {CAI_average_Theta}')
print(f'Average Xi: \nEnKF_AI: {AI_average_Xi}\nEnKF_CAI: {CAI_average_Xi}')
#---------------------------------------------------------------------------------
#--------------------------------------Plotting-----------------------------------
#---------------------------------------------------------------------------------


post_error = evaluate_posterior_error(U, EnKF_values, Time)
post_error_AI = evaluate_posterior_error(U, EnKF_AI_values, Time) 
xaxis = np.arange(0.0, T, h)

plt.figure(figsize=(8, 2.5))    
  
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()  

plt.ylim(0, ymax)


plt.plot(xaxis, post_error, color = (31/255, 119/255, 180/255), lw = 0.5)
plt.plot(xaxis, post_error_AI, color = (255/255, 127/255, 14/255), lw = 0.5)
plt.plot(xaxis, Benchmark_RMSE*np.ones(Time),'--' , color = 'black', lw = 0.3)

plt.gca().legend(('EnKF', 'EnKF-AI'))

plt.show()

post_error_CI = evaluate_posterior_error(U, EnKF_CI_values, Time)
post_error_CAI = evaluate_posterior_error(U, EnKF_CAI_values, Time)

plt.figure(figsize=(8, 2.5))    
  
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()  

plt.ylim(0, ymax)


plt.plot(xaxis, post_error_CI, color = (31/255, 119/255, 180/255), lw = 0.5)
plt.plot(xaxis, post_error_CAI, color = (255/255, 127/255, 14/255), lw = 0.5)
plt.plot(xaxis, Benchmark_RMSE*np.ones(Time),'--' , color = 'black', lw = 0.3)

plt.gca().legend(('EnKF-CI', 'EnKF-CAI'))

plt.show()

#---------------------------------------------------------------------
#----------------------plot of first 3 variables ---------------------
#---------------------------------------------------------------------

# An optinal 3 D visualization using the first 3 components.
'''
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(post_error_CAI[0,:], post_error_CAI[1,:], post_error_CAI[2,:], 'r')
ax.plot(signal[0,:],signal[1,:],signal[2,:])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()

'''