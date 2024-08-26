#Purpose: This script plots the distributions of predicted angles a pretty matrix plot.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 30 March 2018

import matplotlib.pyplot as plt
import numpy as np

from menRot_base_plot import pretty_plot, plot_sem, plot_widths, pretty_colorbar
from menRot_plotGat import _set_ticks

###Define important variables###
ListAnalysis = ['InferResp_TrainAll_TestAll']
ListSelection = 'BothDirSeen' #seen, unseen, etc ...

if ListAnalysis[0] is not 'InferResp_TrainAll_TestAll':
	path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
	res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Figures/'
else:
	path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/' + ListAnalysis[0] + '/GroupRes/Stats/'
	res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/' + ListAnalysis[0] + '/GroupRes/Figures/'
#path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/100ms/' + ListAnalysis[0] + '/GroupRes/Stats/'
#res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/100ms/' + ListAnalysis[0] + '/GroupRes/Figures/'


n_bins = np.linspace(-np.pi, np.pi, 25)
#n_bins = np.linspace(0, np.pi, 13)
times = np.linspace(-0.2, 3.496, 463)
#times = np.linspace(-0.008, 3.392, 18)
#times = np.linspace(-0.104, 3.432, 35)

#Load data
data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy') #subjects x n_bins x time
data2plot = np.array(data2plot)

#Prepare plot
extent = [min(times), max(times), min(n_bins), max(n_bins)]

#Plot
fig, ax = plt.subplots(1, 1, figsize=[8, 3])

#im = ax.matshow(np.mean(data2plot, axis=0), extent=extent, cmap='coolwarm', origin='lower', aspect='auto')
#im = ax.matshow(np.mean(data2plot, axis=0), extent=extent, cmap='coolwarm', origin='lower', aspect='auto', vmin=0.04, vmax=0.07)

#im = ax.contourf(np.mean(data2plot, axis=0), levels=np.linspace(0.048, 0.075, 5), extent=extent, cmap='coolwarm', origin='lower', aspect='equal')
#plt.contour(np.mean(data2plot, axis=0), levels=np.linspace(0.048, 0.075, 5), extent=extent, origin='lower', aspect='equal', linewidths = 0.2, colors = 'k')

im = ax.contourf(np.mean(data2plot, axis=0), levels=np.linspace(0.046, 0.075, 5), extent=extent, cmap='coolwarm', origin='lower', aspect='equal')
plt.contour(np.mean(data2plot, axis=0), levels=np.linspace(0.046, 0.075, 5), extent=extent, origin='lower', aspect='equal', linewidths = 0.2, colors = 'k')


#Add event markers
ax.axvline(0.0, color='k', linewidth=1) #indicates target onset
ax.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
ax.axvline(3.2667, color='k', linewidth=1) #indicates response onset

#Setup ticks
#ax.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
#ax.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

ax.set_yticks(n_bins)
#ax.set_yticklabels([r'$-\pi$', ' ', r'$-\frac{2}{3}\pi$', ' ', ' ', ' ', '0', ' ', ' ', r'$\frac{2}{3}\pi$', ' ', ' ', r'$\pi$'], fontdict={'family': 'arial', 'size': 12})
ax.set_xlim(min(times), max(times))
ax.set_ylim(min(n_bins), max(n_bins))

#pretty_colorbar()

#pretty_plot(ax)

plt.savefig(res_path + ListSelection + '_' + str(len(n_bins)) + '-histogram.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
plt.show()


