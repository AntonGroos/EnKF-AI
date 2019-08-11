#!/usr/bin/python3.7
import sys
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import ENKF as kf
import EnKF_CI as kf_ci
import EnKF_AI as kf_ai
import EnKF_CAI as kf_cai
import threading 
import shutil
from numpy import dot
from tkinter import ttk
from scipy.integrate import odeint
from PIL import ImageTk, Image
from tkinter.messagebox import showinfo

"""
You may have trouble installing the correct PIL library, since the original project was discontinued.
Instead You want to install Pillow, which is still imported via. PIL and contains the same functions.  
"""

# Fonts. 
small, medium, large = ("Arial", 8), ("Arial", 10), ("Arial", 12)


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



# -----------------------------------------------------------------------------------------------------------------------
# The following classes use TKinter to create a user-interface in order to simlify testing and changing of variables.
# -----------------------------------------------------------------------------------------------------------------------


class graphdisplay(tk.Tk):

	"""
	This is the root class, it controls the size of the window and contains all "frames", which are the other windows.
	"""

	def __init__(self, *args, **kwargs):

		tk.Tk.__init__(self, *args, **kwargs)
		tk.Tk.wm_title(self, "EnKF by Anton")

		ws = self.winfo_screenwidth() # width of the screen
		hs = self.winfo_screenheight() # height of the screen

		x = (ws/2) - (450/2)
		y = (hs/2) - (500/2)

		self.geometry('%dx%d+%d+%d' % (450, 500, x, y))

		container =tk.Frame(self)
		container.pack(side= "top", fill= "both", expand = True)
		container.grid_rowconfigure(0, weight = 1)
		container.grid_columnconfigure(0, weight = 1)

		self.last_frame = None
		self.current_frame = start_page
		self.frames = {}

		for F in (start_page, no_inflation, constant_inflation, adaptive_inflation, constant_adaptive_inflation, help_page):

			frame = F(container, self)

			self.frames[F] = frame

			frame.grid(row = 0, column = 0, sticky = "nsew")

		self.show_frame(start_page)

	
	def show_frame(self, cont):

		"""
		show_frames takes the name of a frame as an argument and brings it to the top.
		It also saves the last frame.
		"""

		self.last_frame = self.current_frame
		self.current_frame = cont
		frame = self.frames[cont]
		frame.tkraise() 


class help_page(tk.Frame):

	"""
	This frame contains definitions and information on the variables in text.
	"""

	def __init__(self, parent, controller):

		tk.Frame.__init__(self, parent)

		back_pic = Image.open("backbutton.jpg")
		back_pic = back_pic.resize((20,20), Image.ANTIALIAS)
		self.back_pic_tk = ImageTk.PhotoImage(back_pic)
		home  = ttk.Button(self, image = self.back_pic_tk, command = lambda: controller.show_frame(controller.last_frame))
		home.place(relx = 0, rely = 0, anchor ='nw')

		self.grid_columnconfigure(0, minsize = 30)
		self.grid_rowconfigure([0,9], minsize = 30)

		d_label = ttk.Label(self, text = "d: The system dimension", font = small)
		d_label.grid(row = 1, column = 1, sticky = 'w')

		T_label = ttk.Label(self, text = "T: The time length of the simulation", font = small)
		T_label.grid(row = 2, column = 1, sticky = 'w')

		F_label = ttk.Label(self, text = "F: The constant added in the Lorenz 96 model. Higher values lead to higher turbulence", font = small)
		F_label.grid(row = 3, column = 1, sticky = 'w')

		q_label = ttk.Label(self, text = "q: The number of observed components", font = small)
		q_label.grid(row = 4, column = 1, sticky = 'w')

		delta_label = ttk.Label(self, text = "delta: The stepsize in the numeric integration", font = small)
		delta_label.grid(row = 5, column = 1, sticky = 'w')

		h_label = ttk.Label(self, text = "h: Time between observations", font = small)
		h_label.grid(row = 6, column = 1, sticky = 'w')

		scheme_label = ttk.Label(self, text = "scheme: The numeric interation scheme used in the filter", font = small)
		scheme_label.grid(row = 7, column = 1, sticky ='w')

		K_label = ttk.Label(self, text = "K: The number of ensemble members", font = small)
		K_label.grid(row = 8, column = 1, sticky ='w')

		long_time_label = ttk.Label(self, text = "Long time simulations are made with the current values over 100 trials", font = small)
		long_time_label.grid(row = 10, column = 1)

		long_time_label1 = ttk.Label(self, text = "!Warning!", font = small)
		long_time_label1.grid(row = 11, column = 1)

		long_time_label2 = ttk.Label(self, text = "Long time simulations take considerable calculations and may take up to two hours.", font = small)
		long_time_label2.grid(row = 12, column = 1)



class start_page(tk.Frame):

	"""
	Page where you can select which filter you would like to use.
	"""

	def __init__(self, parent, controller):

		tk.Frame.__init__(self, parent)

		label = ttk.Label(self, text ="Choose Filtering Method", font = large) 
		label.place(relx = 0.5, rely = 0.1, anchor = "n")

		button1 = ttk.Button(self, text ="EnKF", command = lambda: controller.show_frame(no_inflation), width =13)
		button1.place(relx = 0.2, rely = 0.2, anchor = "center")

		button2 = ttk.Button(self, text ="EnKF-CI", command = lambda: controller.show_frame(constant_inflation), width =13)
		button2.place(relx = 0.4, rely = 0.2, anchor = "center")

		button3 = ttk.Button(self, text ="EnKF-AI", command = lambda: controller.show_frame(adaptive_inflation), width =13)
		button3.place(relx = 0.6, rely = 0.2, anchor = "center")

		button4 = ttk.Button(self, text ="EnKF-CAI", command = lambda: controller.show_frame(constant_adaptive_inflation), width =13)
		button4.place(relx = 0.8, rely = 0.2, anchor = "center")

		intro_pic = Image.open("Figure_1.jpg")
		intro_pic = intro_pic.resize((450, 350), Image.ANTIALIAS)
		intro_pic_tk = ImageTk.PhotoImage(intro_pic)
		intro_pic_show = tk.Label(self, image = intro_pic_tk)
		intro_pic_show.image = intro_pic_tk
		intro_pic_show.place(relx = 0.5, rely = 0.6, anchor = "center")


class no_inflation(tk.Frame):

	"""
	Basic Ensemble Kalman-Filter using no type of inflation method.
	--------------------------------------------------------------
	It features following buttons:

	- home takes you back to start_page
	- settings opens a drop down to show advanced options and access to help_page as well as long time simulations
	- Go starts the simulation with current values
	
	It allows the user to enter following values:

	- input_d the system dimension
	- input_T the time length of the simulation
	- input_F the forcing value of the Lorenz 96 model (responsible for turbulence)
	- input_q the number of observed components
	- input_delta the stepsize used in the numerical integration
	- input_h the time between observations
	- input_scheme the intergration scheme used 
		euler = explicit Euler scheme
		runge-kutta = runge-kutta(4)
		odeint = numeric integration function from Scipy library
	- input_K the numer of ensemble members
	"""

	def  __init__(self, parent, controller):
		
		tk.Frame.__init__(self, parent)

		back_pic = Image.open("backbutton.jpg")
		back_pic = back_pic.resize((20,20), Image.ANTIALIAS)
		self.back_pic_tk = ImageTk.PhotoImage(back_pic)
		home  = ttk.Button(self, image = self.back_pic_tk, command = lambda: controller.show_frame(start_page))
		home.place(relx = 0, rely = 0, anchor ='nw')

		settings_pic = Image.open("settings.jpg")
		settings_pic = settings_pic.resize((20, 20), Image.ANTIALIAS)
		self.settings_pic_tk = ImageTk.PhotoImage(settings_pic)
		settings = ttk.Menubutton(self, image = self.settings_pic_tk, direction = 'below')
		settings.place(rely = 0, relx = 1, anchor = 'ne')
		settings.menu = tk.Menu(settings, tearoff = 0)
		settings['menu'] = settings.menu
		settings.menu.add_command(label = 'Help', command = lambda: controller.show_frame(help_page))
		settings.menu.add_command(label = 'Reset defaults', command = lambda: self.reset_defaults())
		settings.menu.add_separator()
		settings.menu.add_command(label = 'Start long time simulation', command = lambda: self.popup_warning('EnKF'))

		#lambda: controller.popup_bonus())

		self.grid_rowconfigure([0,2,8], minsize = 30)
		self.grid_columnconfigure(1, minsize = 70)

		label = ttk.Label(self, text ="EnKF", font = large) 
		label.grid(row = 1, column = 2, columnspan = 3)

		#left column

		self.input_d = ttk.Entry(self)
		self.input_d.insert(0, 5)
		self.input_d.grid(row = 3, column = 2, sticky = 'nsew')
		d_label = ttk.Label(self, text = "d:", font = medium)
		d_label.grid(row = 3, column = 1, sticky = 'e', padx= 5)

		self.input_T = ttk.Entry(self)
		self.input_T.insert(0, 100)
		self.input_T.grid(row = 4, column = 2,  sticky = 'nsew')
		T_label =ttk.Label(self, text = "T:", font = medium)
		T_label.grid(row = 4, column = 1, sticky = 'e', padx= 5)

		self.input_F = tk.IntVar(self)
		self.input_F.set(4)
		F_menu = ttk.OptionMenu(self, self.input_F, 4, 4, 8, 16)
		F_menu.grid(row = 5, column = 2, sticky = 'nsew')
		F_label = ttk.Label(self, text = "F:", font = medium)
		F_label.grid(row = 5, column = 1, sticky = 'e', padx= 5)

		self.input_q = ttk.Entry(self)
		self.input_q.insert(0, 1)
		self.input_q.grid(row = 6, column = 2)
		q_label = ttk.Label(self, text = 'q:', font = medium)
		q_label.grid(row = 6, column = 1, sticky = 'e', padx= 5)

		#right column

		self.input_delta = ttk.Entry(self)
		self.input_delta.insert(0, 0.0001)
		self.input_delta.grid(row = 3, column = 4)
		delta_label = ttk.Label(self, text = 'delta:', font = medium)
		delta_label.grid(row = 3, column = 3, sticky = 'e', padx= 5)


		self.input_h = ttk.Entry(self)
		self.input_h.insert(0, 0.05)
		self.input_h.grid(row = 4, column = 4)
		h_label = ttk.Label(self, text = 'h:', font = medium)
		h_label.grid(row = 4, column = 3, sticky = 'e', padx= 5)

		self.input_scheme = tk.StringVar(self)
		self.input_scheme.set('euler')
		scheme_menu = ttk.OptionMenu(self, self.input_scheme, 'euler', 'euler', 'runge-kutta', 'odeint')
		scheme_menu.grid(row = 5, column = 4, sticky = 'nsew')
		scheme_label = ttk.Label(self, text = "scheme:", font = medium)
		scheme_label.grid(row = 5, column = 3, sticky = 'e', padx= 5)
		self.input_K = ttk.Entry(self)

		self.input_K.insert(0, 6)
		self.input_K.grid(row = 6, column = 4)
		K_label = ttk.Label(self, text = 'K:', font = medium)
		K_label.grid(row = 6, column = 3, sticky = 'e', padx= 5)

		Go = ttk.Button(self, text="Start Simulation",
		command = lambda: self.input_conversion(self.input_d.get(), self.input_T.get(), self.input_delta.get(), 
			self.input_scheme.get(), self.input_q.get(), self.input_K.get(), self.input_F.get(), self.input_h.get(), 'EnKF'))
		
		Go.grid(row = 7, column = 2, columnspan = 3, pady = 5, sticky = 'nsew')



	def reset_defaults(self):

		"""
		This function sets all current values to the original ones.
		"""

		self.input_d.delete(0,'end')
		self.input_d.insert(0,5)
		self.input_K.delete(0,'end')
		self.input_K.insert(0,6)
		self.input_F.set(4)
		self.input_T.delete(0,'end')
		self.input_T.insert(0,100)
		self.input_h.delete(0,'end')
		self.input_h.insert(0,0.05)
		self.input_delta.delete(0,'end')
		self.input_delta.insert(0,0.0001)
		self.input_scheme.set('euler')
		self.input_q.delete(0,'end')
		self.input_q.insert(0,1)

	def popup_warning(self, filter_name):

		'''
		Creates a popup window. With a warning.
		'''

		ws = self.winfo_screenwidth()  # width of the screen
		hs = self.winfo_screenheight() # height of the screen

		x = (ws/2) - (300/2)
		y = (hs/2) - (100/2)

		win = tk.Toplevel()
		win.wm_title("Warning")

		win.geometry('%dx%d+%d+%d' % (300, 100, x, y))

		popup_label = tk.Label(win, text = "This may take up to two hours.\nPress Okay to confirm.")
		popup_label.place(relx = 0.5, rely = 0.3, anchor = "center")

		popup_button = ttk.Button(win, text="Okay", command = lambda:  
			[win.destroy(), self.long_time_simulation(self.input_d.get(), self.input_T.get(), self.input_delta.get(), self.input_scheme.get(), 
			self.input_q.get(), self.input_K.get(), self.input_F.get(), self.input_h.get(), filter_name)])
		popup_button.place(relx = 0.5, rely = 0.7, anchor = "center")


	def input_conversion(self, d, T, delta, scheme, q, K, F, h, filter_name):

		"""
		This function first checks for any inputs that might crash the programm and tells the user if there are any problems.
		Then it coverts the string inputs into numbers that can be used by the respective filters.
		Furthermore it then starts the simulation and creates a plot of the Posterior Error and a ttk table
	 	containing some usefull statistics. It appears only in the no_inflation class,
	 	but can be called in the frames of the other filters, too, because of class inheritance.

	 	First we do some error handeling. Then we compute the remaining variables using the given ones and plug them into 
	 	the declared filter. Finally we plot the result and display some relevant statistics.
		"""

		got_input_error = 0
		error_label = ttk.Label(self)
		error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'nsew')

		try:
			d = int(d)
			if d < 4:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'The Lorenz 96 model needs 4 or more particles.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for d.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			T = int(T)
			if T % 2 != 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'Please enter an even number for T.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
			if T < 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'Please choose T larger than zero.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for T.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			delta = float(delta)
			if delta < 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'delta must be positive.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a number as a value for delta.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			h = float(h)
			if h < 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'h must be positive.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a number as a value for h.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			some_number = int(1/h)
		except ValueError:
			error_label = ttk.Label(self, text = 'Choose a value for h, such that it is a divider of any whole number.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			q = int(q)
			if q <= 0:
				error_label = ttk.Label(self, text = 'Please enter a positive number for q.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for q.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			K = int(K)
			if K <=1:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'K has to be at least 2.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for K.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		if q > d:
			q = d
			error_label = ttk.Label(self, text = 'q exceeds system dimesion and was set to d automaticly.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')

		obs  = 0
		Time = 0

		if (h/delta).is_integer():
			obs = int(h/delta)
		else:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'h/delta has to be a natural number.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		if (T/h).is_integer():
			Time = int(T/h)
			if T == h:
				error_label = ttk.Label(self, text = 'No observations in the choosen time interval.\nConsider increasing T or decreasing h.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		else:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'T/h has to be a natural number.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		if d == q:
			H = np.eye(q)
		else:
			H = np.concatenate((np.eye(q), np.zeros((q,d-q))), axis = 1)

		'''
		Now we create the truth signal using the equilibrium of the different turbulence regimes
		(these where calculated in equilibrium.py) and initiate the choosen filter with the start values.
		'''	

		if got_input_error == 0:

			if F == 4:
				mu, sigma      = 1.22, 3.44   
				Benchmark_RMSE = 3.25	 
				ymax           = 4              	

			elif F == 8:
				mu, sigma      = 2.28, 12.08
				Benchmark_RMSE = 7.02
				ymax           = 25	

			elif F == 16:
				mu, sigma      = 3.1, 44.82
				Benchmark_RMSE = 12.93
				ymax 		   = 80
			else: 
				return		

			measure_noise = (0, 0.01) 								   		
			signal = np.transpose(odeint(kf.Lorenz96, np.random.normal(mu,1,d), np.linspace(0,T,int(T/delta)), args = (F,)))
			U = signal[:,0::obs]	
			observation = np.array([dot(H,U[:,k]) + np.random.normal(measure_noise[0], measure_noise[1], q) for k in range(Time)])

			start = np.random.normal(mu, sigma, (d,K)) 
			start[0:q,:] = np.transpose(np.array([observation[0] for k in range(K)])) + np.random.normal(0, 0.01, (q, K)) 



			if filter_name == 'EnKF':
				start_time = time.time()
				Filter_EV  = kf.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
				end_time   = time.time()-start_time
				RMSE       = evaluate_root_mean_square_error(U,Filter_EV)
				Cor        = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)
				info_display = ttk.Treeview(self, columns=('Numbers'))
				info_display.column('#0', width = 100, anchor = 'w')
				info_display.column('Numbers', width = 100, anchor = 'e')
				info_display.insert('', 'end', 'RMSE', text = 'RMSE', values = round(RMSE, 4))
				info_display.insert('', 'end', 'Pattern_correlation', text = 'Pattern correlation', values = round(Cor, 4))
				info_display.insert('', 'end', 'Time', text = 'Time(seconds)', values = round(end_time, 0))
				info_display.grid(row = 10, column =2, columnspan = 3, sticky = 'nsew')

			elif filter_name == 'EnKF_CI':
				start_time = time.time()
				Filter_EV = kf_ci.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
				end_time   = time.time()-start_time
				RMSE = evaluate_root_mean_square_error(U,Filter_EV)
				Cor  = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)
				info_display = ttk.Treeview(self, columns=('Numbers'))
				info_display.column('#0', width = 100, anchor = 'w')
				info_display.column('Numbers', width = 100, anchor = 'e')
				info_display.insert('', 'end', 'RMSE', text = 'RMSE', values = round(RMSE, 4))
				info_display.insert('', 'end', 'Pattern_correlation', text = 'Pattern correlation', values = round(Cor, 4))
				info_display.insert('', 'end', 'Time', text = 'Time(seconds)', values = round(end_time, 0))
				info_display.grid(row = 10, column =2, columnspan = 3, sticky = 'nsew')

			elif filter_name == 'EnKF_AI':
				start_time = time.time()
				Filter_EV, counter, average_Theta, average_Xi  = kf_ai.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
				end_time   = time.time()-start_time
				RMSE = evaluate_root_mean_square_error(U,Filter_EV)
				Cor  = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)
				info_display = ttk.Treeview(self, columns=('Numbers'))
				info_display.column('#0', width = 100, anchor = 'w')
				info_display.column('Numbers', width = 100, anchor = 'e')
				info_display.insert('', 'end', 'RMSE', text = 'RMSE', values = round(RMSE, 4))
				info_display.insert('', 'end', 'Pattern_correlation', text = 'Pattern correlation', values = round(Cor, 4))
				info_display.insert('', 'end', 'average_Theta', text = 'Average Theta', values = round(average_Theta, 4))
				info_display.insert('', 'end', 'average_Xi', text = 'Average Xi', values = round(average_Xi, 4))
				info_display.insert('', 'end', 'Triggers', text = 'Adap. inflation triggers', values = (counter))
				info_display.insert('', 'end', 'Time', text = 'Time(seconds)', values = round(end_time, 0))
				info_display.grid(row = 10, column =2, columnspan = 3, sticky = 'nsew')

			elif filter_name == 'EnKF_CAI':
				start_time = time.time()
				Filter_EV, counter, average_Theta, average_Xi = kf_cai.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
				end_time   = time.time()-start_time
				RMSE = evaluate_root_mean_square_error(U,Filter_EV)
				Cor  = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)
				info_display = ttk.Treeview(self, columns=('Numbers'))
				info_display.column('#0', width = 100, anchor = 'w')
				info_display.column('Numbers', width = 100, anchor = 'e')
				info_display.insert('', 'end', 'RMSE', text = 'RMSE', values = round(RMSE, 4))
				info_display.insert('', 'end', 'Pattern_correlation', text = 'Pattern correlation', values = round(Cor, 4))
				info_display.insert('', 'end', 'average_Theta', text = 'Average Theta', values = round(average_Theta, 4))
				info_display.insert('', 'end', 'average_Xi', text = 'Average Xi', values = round(average_Xi, 4))
				info_display.insert('', 'end', 'Triggers', text = 'Adap. inflation triggers', values = (counter))
				info_display.insert('', 'end', 'Time', text = 'Time(seconds)', values = round(end_time, 0))
				info_display.grid(row = 10, column =2, columnspan = 3, sticky = 'nsew')
			else: 
				return
			
			#++++++++++++++++++++++++++Plot+++++++++++++++++++++++++++++++++++++++++++++++

			xaxis = np.arange(0.0, T, h)
			post_error = evaluate_posterior_error(U, Filter_EV, Time)

			plt.figure(figsize=(8, 2.5))    
			  
			ax = plt.subplot(111)    
			ax.spines["top"].set_visible(False)    
			ax.spines["bottom"].set_visible(False)    
			ax.spines["right"].set_visible(False)    
			ax.spines["left"].set_visible(False)

			ax.get_xaxis().tick_bottom()    
			ax.get_yaxis().tick_left()  

			plt.ylim(0, ymax)
			plt.xlim(0,T)


			plt.plot(xaxis, post_error, color = (31/255, 119/255, 180/255), lw = 0.5)
			plt.plot(xaxis, Benchmark_RMSE*np.ones(Time),'--' , color = 'black', lw = 0.3)
			plt.gca().legend((filter_name,))
			plt.show()

			return 

	def long_time_simulation(self, d, T, delta, scheme, q, K, F, h, filter_name, N = 100):

		'''
		This function takes all the input values and starts long time simulation of the given Filter.
		It returns a table of all the computed statistics averaged over all trials.
		'''

		got_input_error = 0
		error_label = ttk.Label(self)
		error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'nsew')

		try:
			d = int(d)
			if d < 4:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'The Lorenz 96 model needs 4 or more particles.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for d.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			T = int(T)
			if T % 2 != 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'Please enter an even number for T.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
			if T < 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'Please choose T larger than zero.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for T.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			delta = float(delta)
			if delta < 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'delta must be positive.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a number as a value for delta.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			h = float(h)
			if h < 0:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'h must be positive.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a number as a value for h.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			some_number = int(1/h)
		except ValueError:
			error_label = ttk.Label(self, text = 'Choose a value for h, such that it is a divider of any whole number.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			q = int(q)
			if q <= 0:
				error_label = ttk.Label(self, text = 'Please enter a positive number for q.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for q.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		try:
			K = int(K)
			if K <=1:
				got_input_error = 1
				error_label = ttk.Label(self, text = 'K has to be at least 2.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		except ValueError:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'Please enter a natural number as a value for K.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		if q > d:
			q = d
			error_label = ttk.Label(self, text = 'q exceeds system dimesion and was set to d automaticly.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')

		obs  = 0
		Time = 0

		if (h/delta).is_integer():
			obs = int(h/delta)
		else:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'h/delta has to be a natural number.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		if (T/h).is_integer():
			Time = int(T/h)
			if T == h:
				error_label = ttk.Label(self, text = 'No observations in the choosen time interval.\nConsider increasing T or decreasing h.', font = medium)
				error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
				return
		else:
			got_input_error = 1
			error_label = ttk.Label(self, text = 'T/h has to be a natural number.', font = medium)
			error_label.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')
			return

		if d == q:
			H = np.eye(q)
		else:
			H = np.concatenate((np.eye(q), np.zeros((q,d-q))), axis = 1)

		if got_input_error == 0:

			if F == 4:
				mu, sigma      = 1.22, 3.44   
				Benchmark_RMSE = 3.25	 
				ymax           = 4              	

			elif F == 8:
				mu, sigma      = 2.28, 12.08
				Benchmark_RMSE = 7.02
				ymax           = 25	

			elif F == 16:
				mu, sigma      = 3.1, 44.82
				Benchmark_RMSE = 12.93
				ymax 		   = 80
			else: 
				return		

			measure_noise = (0, 0.01) 								   		
			signal = np.transpose(odeint(kf.Lorenz96, np.random.normal(mu,1,d), np.linspace(0,T,int(T/delta)), args = (F,)))
			U = signal[:,0::obs]	
			
			'''
			The statistics we are collecting.
			'''

			divergencies = 0
			average_RMSE = 0
			average_Cor = 0

			total_average_trigger = 0
			triggered_trials = 0
			total_average_Theta = 0
			total_average_Xi = 0

			progress = ttk.Progressbar(self, orient = 'horizontal', length=100, mode = 'determinate')
			progress.grid(row = 10, column = 2, columnspan = 3, sticky = 'n')

			for n in range(N):

				progress['value'] = n+1
				self.update_idletasks()

				observation = np.array([dot(H,U[:,k]) + np.random.normal(measure_noise[0], measure_noise[1], q) for k in range(Time)])

				start = np.random.normal(mu, sigma, (d,K))  
				start[0:q,:] = np.transpose(np.array([observation[0] for k in range(K)])) + np.random.normal(0, 0.01, (q, K))

				if filter_name == 'EnKF':

					start_time = time.time()
					Filter_EV  = kf.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
					end_time   = time.time()-start_time
					RMSE       = evaluate_root_mean_square_error(U,Filter_EV)
					Cor        = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)

					if np.isnan(RMSE):

						divergencies += 1

					average_RMSE += 1/N*RMSE
					average_Cor += 1/N*Cor

				elif filter_name == 'EnKF_CI':

					start_time = time.time()
					Filter_EV = kf_ci.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
					end_time   = time.time()-start_time
					RMSE = evaluate_root_mean_square_error(U,Filter_EV)
					Cor  = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)

					if np.isnan(RMSE):

						divergencies += 1

					average_RMSE += 1/N*RMSE
					average_Cor += 1/N*Cor

				elif filter_name == 'EnKF_AI':

					start_time = time.time()
					Filter_EV, counter, average_Theta, average_Xi  = kf_ai.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
					end_time   = time.time()-start_time
					RMSE = evaluate_root_mean_square_error(U,Filter_EV)
					Cor  = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)

					if np.isnan(RMSE):

						divergencies += 1

					average_RMSE += 1/N*RMSE
					average_Cor += 1/N*Cor

					total_average_trigger += 1/N*counter
					total_average_Xi += average_Xi
					total_average_Theta += average_Theta

					if counter > 0:

						triggered_trials += 1


				elif filter_name == 'EnKF_CAI':

					start_time = time.time()
					Filter_EV, counter, average_Theta, average_Xi = kf_cai.my_EnKF(observation, H, start, measure_noise, Time, K, d, q, delta, scheme, obs, F)
					end_time   = time.time()-start_time
					RMSE = evaluate_root_mean_square_error(U,Filter_EV)
					Cor  = evaluate_pattern_correlation(U, Filter_EV, T, mu*np.ones(d), h)

					if np.isnan(RMSE):

						divergencies += 1

					average_RMSE += 1/N*RMSE
					average_Cor += 1/N*Cor

					total_average_trigger += 1/N*counter
					total_average_Xi += average_Xi
					total_average_Theta += average_Theta

					if counter > 0:

						triggered_trials += 1


				else: 

					return


			info_display = ttk.Treeview(self, columns=('Numbers'))
			info_display.column('#0', width = 100, anchor = 'w')
			info_display.column('Numbers', width = 100, anchor = 'e')
			info_display.insert('', 'end', 'Divergencies', text = 'Divergencies', values = divergencies)
			info_display.insert('', 'end', 'RMSE', text = 'Avrg. RMSE', values = round(average_RMSE, 4))
			info_display.insert('', 'end', 'Pattern_correlation', text = 'Avrg. pattern cor.', values = round(average_Cor, 4))

			if filter_name == 'EnKF_CAI' or filter_name == 'EnKF_AI':

				info_display.insert('', 'end', 'average_Theta', text = 'Average Theta', values = round(total_average_Theta, 4))
				info_display.insert('', 'end', 'average_Xi', text = 'Average Xi', values = round(total_average_Xi, 4))
				info_display.insert('', 'end', 'Triggers', text = 'Avrg. adap. infl. triggers', values = round(total_average_trigger, 2))
				info_display.insert('', 'end', 'Triggered_trials', text = 'Trials with trig. adap. inf.', values = round(end_time, 0))

			info_display.grid(row = 10, column =2, columnspan = 3, sticky = 'nsew')


class constant_inflation(no_inflation):

	"""
	Basic Ensemble Kalman-Filter using constant additive covariance inflation.
	--------------------------------------------------------------
	It features following buttons:

	- home takes you back to start_page
	- settings opens a drop down to show advanced options and access to help_page as well as long time simulations
	- Go starts the simulation with current values
	
	It allows the user to enter following values:

	- input_d the system dimension
	- input_T the time length of the simulation
	- input_F the forcing value of the Lorenz 96 model (responsible for turbulence)
	- input_q the number of observed components
	- input_delta the stepsize used in the numerical integration
	- input_h the time between observations
	- input_scheme the intergration scheme used 
		euler = explicit Euler scheme
		runge-kutta = runge-kutta(4)
		odeint = numeric integration function from Scipy library
	- input_K the numer of ensemble members
	"""

	def  __init__(self, parent, controller):
		
		tk.Frame.__init__(self, parent)

		back_pic = Image.open("backbutton.jpg")
		back_pic = back_pic.resize((20, 20), Image.ANTIALIAS)
		self.back_pic_tk = ImageTk.PhotoImage(back_pic)
		home  = ttk.Button(self, image = self.back_pic_tk, command = lambda: controller.show_frame(start_page))
		home.place(relx = 0, rely = 0, anchor ='nw')

		settings_pic = Image.open("settings.jpg")
		settings_pic = settings_pic.resize((20, 20), Image.ANTIALIAS)
		self.settings_pic_tk = ImageTk.PhotoImage(settings_pic)
		settings = ttk.Menubutton(self, image = self.settings_pic_tk, direction = 'below')
		settings.place(rely = 0, relx = 1, anchor = 'ne')
		settings.menu = tk.Menu(settings, tearoff = 0)
		settings['menu'] = settings.menu
		settings.menu.add_command(label = 'Help', command = lambda: controller.show_frame(help_page))
		settings.menu.add_command(label = 'Reset defaults', command = lambda: self.reset_defaults())
		settings.menu.add_separator()
		settings.menu.add_command(label = 'Start long time simulation', command = lambda: self.popup_warning('EnKF_CI'))

		self.grid_rowconfigure([0, 2, 8], minsize = 30)
		self.grid_columnconfigure(1, minsize = 70)

		label = ttk.Label(self, text ="EnKF-CI", font = large) 
		label.grid(row = 1, column = 2, columnspan = 3)

		#left column

		self.input_d = ttk.Entry(self)
		self.input_d.insert(0, 5)
		self.input_d.grid(row = 3, column = 2, sticky = 'nsew')
		d_label = ttk.Label(self, text = "d:", font = medium)
		d_label.grid(row = 3, column = 1, sticky = 'e', padx= 5)

		self.input_T = ttk.Entry(self)
		self.input_T.insert(0, 100)
		self.input_T.grid(row = 4, column = 2,  sticky = 'nsew')
		T_label =ttk.Label(self, text = "T:", font = medium)
		T_label.grid(row = 4, column = 1, sticky = 'e', padx= 5)

		self.input_F = tk.IntVar(self)
		self.input_F.set(4)
		F_menu = ttk.OptionMenu(self, self.input_F, 4, 4, 8, 16)
		F_menu.grid(row = 5, column = 2, sticky = 'nsew')
		F_label = ttk.Label(self, text = "F:", font = medium)
		F_label.grid(row = 5, column = 1, sticky = 'e', padx= 5)

		self.input_q = ttk.Entry(self)
		self.input_q.insert(0, 1)
		self.input_q.grid(row = 6, column = 2)
		q_label = ttk.Label(self, text = 'q:', font = medium)
		q_label.grid(row = 6, column = 1, sticky = 'e', padx= 5)

		#right column

		self.input_delta = ttk.Entry(self)
		self.input_delta.insert(0, 0.0001)
		self.input_delta.grid(row = 3, column = 4)
		delta_label = ttk.Label(self, text = 'delta:', font = medium)
		delta_label.grid(row = 3, column = 3, sticky = 'e', padx= 5)


		self.input_h = ttk.Entry(self)
		self.input_h.insert(0, 0.05)
		self.input_h.grid(row = 4, column = 4)
		h_label = ttk.Label(self, text = 'h:', font = medium)
		h_label.grid(row = 4, column = 3, sticky = 'e', padx= 5)

		self.input_scheme = tk.StringVar(self)
		self.input_scheme.set('euler')
		scheme_menu = ttk.OptionMenu(self, self.input_scheme, 'euler', 'euler', 'runge-kutta', 'odeint')
		scheme_menu.grid(row = 5, column = 4, sticky = 'nsew')
		scheme_label = ttk.Label(self, text = "scheme:", font = medium)
		scheme_label.grid(row = 5, column = 3, sticky = 'e', padx= 5)
		self.input_K = ttk.Entry(self)

		self.input_K.insert(0, 6)
		self.input_K.grid(row = 6, column = 4)
		K_label = ttk.Label(self, text = 'K:', font = medium)
		K_label.grid(row = 6, column = 3, sticky = 'e', padx= 5)

		Go = ttk.Button(self, text="Start Simulation", 
		command = lambda: self.input_conversion(self.input_d.get(), self.input_T.get(), self.input_delta.get(), self.input_scheme.get(), self.input_q.get(), self.input_K.get(), self.input_F.get(), self.input_h.get(), 'EnKF_CI'))
		Go.grid(row = 7, column = 2, columnspan = 3, pady = 5, sticky = 'nsew')


class adaptive_inflation(no_inflation):

	"""
	Basic Ensemble Kalman-Filter using adaptive additive covariance inflation.
	--------------------------------------------------------------
	It features following buttons:

	- home takes you back to start_page
	- settings opens a drop down to show advanced options and access to help_page as well as long time simulations
	- Go starts the simulation with current values
	
	It allows the user to enter following values:

	- input_d the system dimension
	- input_T the time length of the simulation
	- input_F the forcing value of the Lorenz 96 model (responsible for turbulence)
	- input_q the number of observed components
	- input_delta the stepsize used in the numerical integration
	- input_h the time between observations
	- input_scheme the intergration scheme used 
		euler = explicit Euler scheme
		runge-kutta = runge-kutta(4)
		odeint = numeric integration function from Scipy library
	- input_K the numer of ensemble members
	"""

	def  __init__(self, parent, controller):
		
		tk.Frame.__init__(self, parent)

		back_pic = Image.open("backbutton.jpg")
		back_pic = back_pic.resize((20, 20), Image.ANTIALIAS)
		self.back_pic_tk = ImageTk.PhotoImage(back_pic)
		home  = ttk.Button(self, image = self.back_pic_tk, command = lambda: controller.show_frame(start_page))
		home.place(relx = 0, rely = 0, anchor ='nw')

		settings_pic = Image.open("settings.jpg")
		settings_pic = settings_pic.resize((20, 20), Image.ANTIALIAS)
		self.settings_pic_tk = ImageTk.PhotoImage(settings_pic)
		settings = ttk.Menubutton(self, image = self.settings_pic_tk, direction = 'below')
		settings.place(rely = 0, relx = 1, anchor = 'ne')
		settings.menu = tk.Menu(settings, tearoff = 0)
		settings['menu'] = settings.menu
		settings.menu.add_command(label = 'Help', command = lambda: controller.show_frame(help_page))
		settings.menu.add_command(label = 'Reset defaults', command = lambda: self.reset_defaults())
		settings.menu.add_separator()
		settings.menu.add_command(label = 'Start long time simulation', command = lambda: self.popup_warning('EnKF_AI'))

		self.grid_rowconfigure([0, 2, 8], minsize = 30)
		self.grid_columnconfigure(1, minsize = 70)

		label = ttk.Label(self, text ="EnKF-AI", font = large) 
		label.grid(row = 1, column = 2, columnspan = 3)

		#left column

		self.input_d = ttk.Entry(self)
		self.input_d.insert(0, 5)
		self.input_d.grid(row = 3, column = 2, sticky = 'nsew')
		d_label = ttk.Label(self, text = "d:", font = medium)
		d_label.grid(row = 3, column = 1, sticky = 'e', padx= 5)

		self.input_T = ttk.Entry(self)
		self.input_T.insert(0, 100)
		self.input_T.grid(row = 4, column = 2,  sticky = 'nsew')
		T_label =ttk.Label(self, text = "T:", font = medium)
		T_label.grid(row = 4, column = 1, sticky = 'e', padx= 5)

		self.input_F = tk.IntVar(self)
		self.input_F.set(4)
		F_menu = ttk.OptionMenu(self, self.input_F, 4, 4, 8, 16)
		F_menu.grid(row = 5, column = 2, sticky = 'nsew')
		F_label = ttk.Label(self, text = "F:", font = medium)
		F_label.grid(row = 5, column = 1, sticky = 'e', padx= 5)

		self.input_q = ttk.Entry(self)
		self.input_q.insert(0, 1)
		self.input_q.grid(row = 6, column = 2)
		q_label = ttk.Label(self, text = 'q:', font = medium)
		q_label.grid(row = 6, column = 1, sticky = 'e', padx= 5)

		#right column

		self.input_delta = ttk.Entry(self)
		self.input_delta.insert(0, 0.0001)
		self.input_delta.grid(row = 3, column = 4)
		delta_label = ttk.Label(self, text = 'delta:', font = medium)
		delta_label.grid(row = 3, column = 3, sticky = 'e', padx= 5)


		self.input_h = ttk.Entry(self)
		self.input_h.insert(0, 0.05)
		self.input_h.grid(row = 4, column = 4)
		h_label = ttk.Label(self, text = 'h:', font = medium)
		h_label.grid(row = 4, column = 3, sticky = 'e', padx= 5)

		self.input_scheme = tk.StringVar(self)
		self.input_scheme.set('euler')
		scheme_menu = ttk.OptionMenu(self, self.input_scheme, 'euler', 'euler', 'runge-kutta', 'odeint')
		scheme_menu.grid(row = 5, column = 4, sticky = 'nsew')
		scheme_label = ttk.Label(self, text = "scheme:", font = medium)
		scheme_label.grid(row = 5, column = 3, sticky = 'e', padx= 5)
		self.input_K = ttk.Entry(self)

		self.input_K.insert(0, 6)
		self.input_K.grid(row = 6, column = 4)
		K_label = ttk.Label(self, text = 'K:', font = medium)
		K_label.grid(row = 6, column = 3, sticky = 'e', padx= 5)

		Go = ttk.Button(self, text="Start Simulation", 
		command = lambda: self.input_conversion(self.input_d.get(), self.input_T.get(), self.input_delta.get(), self.input_scheme.get(), self.input_q.get(), self.input_K.get(), self.input_F.get(), self.input_h.get(), 'EnKF_AI'))
		Go.grid(row = 7, column = 2, columnspan = 3, pady = 5, sticky = 'nsew')


class constant_adaptive_inflation(no_inflation):

	"""
	Basic Ensemble Kalman-Filter using  constant and adaptive additive covariance inflation
	--------------------------------------------------------------
	It features following buttons:

	- home takes you back to start_page
	- settings opens a drop down to show advanced options and access to help_page as well as long time simulations
	- Go starts the simulation with current values
	
	It allows the user to enter following values:

	- input_d the system dimension
	- input_T the time length of the simulation
	- input_F the forcing value of the Lorenz 96 model (responsible for turbulence)
	- input_q the number of observed components
	- input_delta the stepsize used in the numerical integration
	- input_h the time between observations
	- input_scheme the intergration scheme used 
		euler = explicit Euler scheme
		runge-kutta = runge-kutta(4)
		odeint = numeric integration function from Scipy library
	- input_K the numer of ensemble members
	"""

	def  __init__(self, parent, controller):
		
		tk.Frame.__init__(self, parent)

		back_pic = Image.open("backbutton.jpg")
		back_pic = back_pic.resize((20, 20), Image.ANTIALIAS)
		self.back_pic_tk = ImageTk.PhotoImage(back_pic)
		home  = ttk.Button(self, image = self.back_pic_tk, command = lambda: controller.show_frame(start_page))
		home.place(relx = 0, rely = 0, anchor ='nw')

		settings_pic = Image.open("settings.jpg")
		settings_pic = settings_pic.resize((20, 20), Image.ANTIALIAS)
		self.settings_pic_tk = ImageTk.PhotoImage(settings_pic)
		settings = ttk.Menubutton(self, image = self.settings_pic_tk, direction = 'below')
		settings.place(rely = 0, relx = 1, anchor = 'ne')
		settings.menu = tk.Menu(settings, tearoff = 0)
		settings['menu'] = settings.menu
		settings.menu.add_command(label = 'Help', command = lambda: controller.show_frame(help_page))
		settings.menu.add_command(label = 'Reset defaults', command = lambda: self.reset_defaults())
		settings.menu.add_separator()
		settings.menu.add_command(label = 'Start long time simulation', command = lambda: self.popup_warning('EnKF_CAI'))

		self.grid_rowconfigure([0, 2, 8], minsize = 30)
		self.grid_columnconfigure(1, minsize = 70)

		label = ttk.Label(self, text ="EnKF-CAI", font = large) 
		label.grid(row = 1, column = 2, columnspan = 3)

		#left column

		self.input_d = ttk.Entry(self)
		self.input_d.insert(0, 5)
		self.input_d.grid(row = 3, column = 2, sticky = 'nsew')
		d_label = ttk.Label(self, text = "d:", font = medium)
		d_label.grid(row = 3, column = 1, sticky = 'e', padx= 5)

		self.input_T = ttk.Entry(self)
		self.input_T.insert(0, 100)
		self.input_T.grid(row = 4, column = 2,  sticky = 'nsew')
		T_label =ttk.Label(self, text = "T:", font = medium)
		T_label.grid(row = 4, column = 1, sticky = 'e', padx= 5)

		self.input_F = tk.IntVar(self)
		self.input_F.set(4)
		F_menu = ttk.OptionMenu(self, self.input_F, 4, 4, 8, 16)
		F_menu.grid(row = 5, column = 2, sticky = 'nsew')
		F_label = ttk.Label(self, text = "F:", font = medium)
		F_label.grid(row = 5, column = 1, sticky = 'e', padx= 5)

		self.input_q = ttk.Entry(self)
		self.input_q.insert(0, 1)
		self.input_q.grid(row = 6, column = 2)
		q_label = ttk.Label(self, text = 'q:', font = medium)
		q_label.grid(row = 6, column = 1, sticky = 'e', padx= 5)

		#right column

		self.input_delta = ttk.Entry(self)
		self.input_delta.insert(0, 0.0001)
		self.input_delta.grid(row = 3, column = 4)
		delta_label = ttk.Label(self, text = 'delta:', font = medium)
		delta_label.grid(row = 3, column = 3, sticky = 'e', padx= 5)


		self.input_h = ttk.Entry(self)
		self.input_h.insert(0, 0.05)
		self.input_h.grid(row = 4, column = 4)
		h_label = ttk.Label(self, text = 'h:', font = medium)
		h_label.grid(row = 4, column = 3, sticky = 'e', padx= 5)

		self.input_scheme = tk.StringVar(self)
		self.input_scheme.set('euler')
		scheme_menu = ttk.OptionMenu(self, self.input_scheme, 'euler', 'euler', 'runge-kutta', 'odeint')
		scheme_menu.grid(row = 5, column = 4, sticky = 'nsew')
		scheme_label = ttk.Label(self, text = "scheme:", font = medium)
		scheme_label.grid(row = 5, column = 3, sticky = 'e', padx= 5)
		self.input_K = ttk.Entry(self)

		self.input_K.insert(0, 6)
		self.input_K.grid(row = 6, column = 4)
		K_label = ttk.Label(self, text = 'K:', font = medium)
		K_label.grid(row = 6, column = 3, sticky = 'e', padx= 5)

		Go = ttk.Button(self, text="Start Simulation", 
		command = lambda: self.input_conversion(self.input_d.get(), self.input_T.get(), self.input_delta.get(), self.input_scheme.get(), self.input_q.get(), self.input_K.get(), self.input_F.get(), self.input_h.get(), 'EnKF_CAI'))
		Go.grid(row = 7, column = 2, columnspan = 3, pady = 5, sticky = 'nsew')



if __name__ == '__main__':
	app = graphdisplay()
	app. mainloop()



