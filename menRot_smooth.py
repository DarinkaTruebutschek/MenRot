#Purpose: Smooth data by averaging.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 19 December 2017

import numpy as np

from scipy.ndimage.filters import generic_filter as gf

def my_smooth(data, window):
	
	data = np.array(data)

	print('Dimension of data to smooth: ' + str(data.ndim))
	
	#Check whether data is 1D (diagonal) or 2D (GAT)
	if data.ndim == 1:
		for t in range(data.shape[0]): #loop through the entire dataset
			if t <= window: #beginning of data
				data[t] = np.mean(data[t : (t + window + 1)])
			elif t >= data.shape[0] - window: #end of data
				data[t] = np.mean(data[(t - window) : t + 1])
			else:
				data[t] = np.mean(data[(t - window) : (t + window + 1)])
		
	elif data.ndim == 2: #Gat matrix
		kernel = np.ones((2 * window + 1, 2 * window + 1))
		data = gf(data, np.mean, footprint=kernel)
	return data