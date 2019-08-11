#!/usr/bin/python3.7
import sys
import csv 
import numpy as np

'''
Here 70000 = (6+1)*2000*5 = (K+1)*(T/h)*d and 100 = N.
If You want to change some values out of plotfilters.py 
and still have it work pick 

A = np.random.normal('new error mean', 'new error variance', ((K+1)*(T/h)*d,N)).
'''

A = np.random.normal(0, 0.01, (70000, 100))

f = open('Error_matrix.txt', 'w+')
	

f.write('\n'.join([','.join([str(cell) for cell in row]) for row in A]))

f.close()
