#Purpose: Pretty bar plot.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 12 March 2018

###Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap
from menRot_base_plot import pretty_plot, plot_sem, plot_widths, pretty_colorbar

def plotBar(data, sem, p_values, my_col):

	data = np.squeeze(data)
	sem = np.squeeze(sem)
	p_values = np.squeeze(p_values)

	#Determine how many bars are going to be needed
	n_bars = len(data)
	pos = np.divide(np.arange(n_bars) + 1, 2.)

	#Plot
	if np.shape(data)[0] == 6:
		fig, ax = plt.subplots(1, figsize = [4,4])
	else:
		fig, ax = plt.subplots(1, figsize = [8,8])

	for toi_i in np.arange(n_bars):
		ax.bar(pos[toi_i], data[toi_i], width = 0.25, align = 'center', linewidth = 5, color = 'w', edgecolor = my_col, 
			yerr = sem[toi_i], error_kw = {'elinewidth': 3}, ecolor = my_col)

		plt.hold(True)

		#Plot significance
		if (p_values[toi_i] < .05) & (p_values[toi_i] > .01):
			plt.scatter(pos[toi_i], (data[toi_i] + sem[toi_i] + 0.003), marker = '*', c = 'b', s = 20)
		elif (p_values[toi_i] < .01) & (p_values[toi_i] > .001):
			plt.scatter((pos[toi_i] -  0.1), (data[toi_i] + sem[toi_i] + 0.003), marker = '*', c = 'b', s = 20)
			plt.scatter((pos[toi_i] +  0.1), (data[toi_i] + sem[toi_i] + 0.003), marker = '*', c = 'b', s = 20)
		elif p_values[toi_i] < .001:
			plt.scatter((pos[toi_i] - 0.1), (data[toi_i] + sem[toi_i] + 0.003), marker = '*', c = 'b', s = 20)
			plt.scatter(pos[toi_i], (data[toi_i] + sem[toi_i] + 0.003), marker = '*', c = 'b', s = 20)
			plt.scatter((pos[toi_i] + 0.1), (data[toi_i] + sem[toi_i] + 0.003), marker = '*', c = 'b', s = 20)
	if np.max(data) > 0.5:
		ax.set_ylim([0.5, 0.61])
	else:
		[ymin, ymax] = ax.set_ylim([np.min(data)-sem[np.argmin(data)]-0.01, np.max(data)+sem[np.argmax(data)]+0.01])
		ax.axhline(0, color='dimgray', linewidth=1, linestyle='dotted')


	ax.set_xticks(pos)
	if np.shape(data)[0] == 6:
		ax.set_xticklabels(['Bl', 'Early', 'P3b', 'Del1', 'Del2', 'Resp'], fontdict={'family': 'arial', 'size': 12})
	else:
		ax.set_xticklabels(['1.82', '1.92', '2.02', '2.12', '2.22', '2.32', '2.42', '2.52', '2.62', '2.72', '2.82', '2.92', '3.02', '3.12', '3.22'], fontdict={'family': 'arial', 'size': 12})

	if np.max(data) > 0.5:
		ax.set_yticks([0.5, 0.61])
		ax.set_yticklabels(['chance', '0.61'], fontdict={'family': 'arial', 'size': 12})
	else:
		ax.set_yticks([0, ymax])
		ax.set_yticklabels(['chance', str(np.round(ymax, 2))], fontdict={'family': 'arial', 'size': 12})

	#ax.set_aspect('equal')

	pretty_plot(ax)

	return ax
