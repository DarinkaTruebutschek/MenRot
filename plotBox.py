#Purpose: Pretty box plot.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 02 May 2018

###Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap
from menRot_base_plot import pretty_plot, plot_sem, plot_widths, pretty_colorbar

def plotBox(data, sem, p_values, my_col, ymax, ymin):

	data = np.squeeze(data)
	p_values = np.squeeze(p_values)

	#Determine how many bars are going to be needed
	n_bars = np.shape(data)[1]
	pos = (np.divide(np.arange(n_bars) + 1, 1))
	#pos = [[1], [2], [3], [4], [5], [6]]

	#Plot
	if (np.shape(data)[1] == 6) or (np.shape(data)[1] == 7) or (np.shape(data)[1] == 5):
		fig, ax = plt.subplots(1, figsize = [3,4]) #[4,4]
	else:
		fig, ax = plt.subplots(1, figsize = [8,8])

	plt.hold(True)
	bp = ax.boxplot(data, showfliers=0, showcaps=0)

	#Prettify
	plt.setp(bp['boxes'], color=my_col, linewidth=3)
	plt.setp(bp['whiskers'], color=my_col, linewidth=2, linestyle='-')
	plt.setp(bp['medians'], color='white', linewidth=2)

	#Fill the boxes with the desired color
	for toi_i in np.arange(n_bars):
		box = bp['boxes'][toi_i]
		boxX = []
		boxY = []
		
		for j in range(5):
			boxX.append(box.get_xdata()[j])
			boxY.append(box.get_ydata()[j])
		boxCoords = np.column_stack([boxX, boxY])

		boxPolygon = plt.Polygon(boxCoords, facecolor=my_col, alpha=.3)
		ax.add_patch(boxPolygon)

	#Overlay individual scores from subjects
	for toi_i in np.arange(n_bars):
		for subi in np.arange(len(data)):
			plt.scatter(pos[toi_i], data[subi][toi_i], c=my_col, s=15, edgecolor=my_col, alpha=.5)


		#Plot significance
		if (p_values[toi_i] < .05) & (p_values[toi_i] > .01):
			plt.scatter(pos[toi_i], (np.max(data, axis=0)[toi_i] + 0.2), marker = '*', c = 'b', s = 20)
		elif (p_values[toi_i] < .01) & (p_values[toi_i] > .001):
			plt.scatter((pos[toi_i] -  0.1), (np.max(data, axis=0)[toi_i] + 0.2), marker = '*', c = 'b', s = 20)
			plt.scatter((pos[toi_i] +  0.1), (np.max(data, axis=0)[toi_i] + 0.2), marker = '*', c = 'b', s = 20)
		elif p_values[toi_i] < .001:
			plt.scatter((pos[toi_i] - 0.1), (np.max(data, axis=0)[toi_i] + 0.2), marker = '*', c = 'b', s = 20)
			plt.scatter(pos[toi_i], (np.max(data, axis=0)[toi_i] + 0.2), marker = '*', c = 'b', s = 20)
			plt.scatter((pos[toi_i] + 0.1), (np.max(data, axis=0)[toi_i] + 0.2), marker = '*', c = 'b', s = 20)

	if ymin is None:
		[ymin, ymax] = ax.set_ylim([np.min(data)-0.05, ymax])
	elif ymax is None:
		[ymin, ymax] = ax.set_ylim([np.min(data)-0.05, np.max(data)+0.1])
	else:
		[ymin, ymax] = ax.set_ylim([ymin, ymax])
	ax.axhline(0, color='dimgray', linewidth=1, linestyle='dotted')
	ax.axhline(np.deg2rad(-120), color='dimgray', linewidth=1, linestyle='dotted')
	ax.axhline(np.deg2rad(120), color='dimgray', linewidth=1, linestyle='dotted')


	#ax.set_xticks(pos)
	if np.shape(data)[1] == 6:
		ax.set_xticklabels(['Bl', 'E', 'P3b', 'D1', 'D2', 'R'], fontdict={'family': 'arial', 'size': 13})
	elif np.shape(data)[1] == 5:
		ax.set_xticklabels(['E', 'P3b', 'D1', 'D2', 'R'], fontdict={'family': 'arial', 'size': 13})
	elif np.shape(data)[1] == 7:
		ax.set_xticklabels(['Bl', 'Early', 'P3b', 'Del1', 'Del2', 'Del3', 'Resp'], fontdict={'family': 'arial', 'size': 13})
	else:
		ax.set_xticklabels(['1.82', '1.92', '2.02', '2.12', '2.22', '2.32', '2.42', '2.52', '2.62', '2.72', '2.82', '2.92', '3.02', '3.12', '3.22'], fontdict={'family': 'arial', 'size': 13})

	ax.set_yticks([np.deg2rad(-120), 0, np.deg2rad(120)])
	#ax.set_yticklabels([str(np.round(ymin, 2)), '0', str(np.round(ymax, 2))], fontdict={'family': 'arial', 'size': 13})
	ax.set_yticklabels(['+120', 'T', '-120'], fontdict={'family': 'arial', 'size': 13})

	pretty_plot(ax)

	return ax