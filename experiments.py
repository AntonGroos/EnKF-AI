#!/usr/bin/python3.7
import sys
import csv 
import numpy as np


A = np.random.normal(0, 0.01, (70000, 100))

f = open('Error_matrix.txt', 'w+')
	

f.write('\n'.join([','.join([str(cell) for cell in row]) for row in A]))

f.close()