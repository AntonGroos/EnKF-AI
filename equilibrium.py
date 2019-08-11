#!/usr/bin/python3.7
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

'''
This programm helps you calculate equilibrium mean and variance, as well as the performance threshholds.
The variables are:

- F The turbulence regime
- N the number of trials run
- K the number of ensemble members(only used for threshholds)
- d the system dimension
- q the number of observed components
- T the total time run in each test(keep it large to guarantee that the system has stabilized)
- size the step size of the numerical integrator

With the default vaules one run of this programm should take approximatly 15 min. 
'''

F = 8
N = 100
K = 6
d = 5
q = 1
T = 10**4
size = 0.1
number = int(T/size)



#print(np.sqrt(np.linalg.norm(H, ord =2)*12.93 + 2*d))

def Lorenz96(x0, t, F):
  
  d =  len(x0)
  x = np.zeros(d)

  x[0] = (x0[1] - x0[d-2]) * x0[d-1] -x0[0] + F 
  x[1] = (x0[2] - x0[d-1]) * x0[0] - x0[1] + F
  x[d-1] = (x0[0] - x0[d-3]) * x0[d-2] - x0[d-1] + F

  for j in range(2,d-1):
    x[j] = (x0[j+1] - x0[j-2]) * x0[j-1] - x0[j] + F
  return x


if d == q:
	H = np.eye(q)
else:
	H = np.concatenate((np.eye(q), np.zeros((q,d-q))), axis = 1)



length = np.linspace(0, T, number)
variable = np.zeros((d,N))

for n in range(N):

	x = np.random.normal(F, 10, d)
	print('still going', n)
	variable[:,n] =  odeint(Lorenz96,x,length, args = (F,))[-1,:]

#print(np.cov(variable), 'cov(U)')

print(1/N * np.sum(variable, axis = 1), 'E[U]')

print(1/(N*d)*np.sum(variable, axis = (0,1)), 'E[x_i]')

covariance_U = np.cov(variable)

covariance_r = covariance_U - np.dot(covariance_U, np.dot(np.transpose(H), np.dot(np.linalg.inv(np.eye(q) + np.dot(H, np.dot(covariance_U, np.transpose(H)))), np.dot(H,covariance_U))))

#print(covariance_r, 'cov(r_k)')

print(np.trace(covariance_r), 'spur cov(r_k)')
print(np.trace(covariance_U)/d, 'spur cov(U)/d = var(x_i)')

M_1 = np.sqrt(np.linalg.norm(H, ord = 2)*np.trace(covariance_r) + 2*d)
M_2 = K/(2*K-2)*np.trace(covariance_r)

M_1_squared = M_1**2

print(f'From this we can calculate the threshholds M_1 = {M_1} and M_2 = {M_2} as well as for the algorithm (M_1)^2 = {M_1_squared}.')