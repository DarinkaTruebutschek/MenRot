#Purpose: This script plots the distributions of predicted angles.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 15 May 2018

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import interpolate
from scipy import ndimage

from menRot_base_plot import pretty_plot, plot_sem, plot_widths, pretty_colorbar
from menRot_plotGat import _set_ticks
from menRot_smooth import my_smooth

###Define important variables###
ListAnalysis = ['Resp_TrainAll_TestAll']
ListSelection = 'BothDirSeen' #'BothDirUnseen' #seen, unseen, etc ...
ListTois = [[0.1, 0.3], [0.3, 0.6], [0.6, 1.768], [1.768, 2.068], [2.068, 2.368], [2.368, 2.668], [2.668, 2.968], [2.968, 3.268], [3.268, 3.5]]
ListTois = [[0.3, 0.4], [2.5, 2.6], [2.8, 2.9], [3.1, 3.2]]
ListTois = [[0.5, 0.6], [1.0, 1.1], [1.5, 1.6], [2.0, 2.1], [2.5, 2.6], [3.0, 3.1]]
#ListTois = [[0.1, 0.3], [0.3, 0.6]]

smooth = False
smoothWindow = 40

interpol=True
baselineCorrection = False

n_bins = np.linspace(-np.pi, np.pi, 25)
my_xTicks = [np.mean(n_bins[current : current+2]) for current in np.arange(len(n_bins))]
my_xTicks = np.rad2deg(my_xTicks[0 :-1])
times = np.linspace(-0.2, 3.496, 463)

#Define colors
my_colors = sns.cubehelix_palette(len(ListTois))

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Figures/'

#Load data
data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy') #subjects x n_bins x time

#Determine whether to baseline correct-data
if baselineCorrection:
	data2plot_corr = np.zeros_like(data2plot)
	baselinePeriod1 = np.abs(times-(-0.2)).argmin()
	baselinePeriod2 = np.abs(times-(-0.0)).argmin()
	for subi in np.arange(np.shape(data2plot)[0]):
		tmp = np.mean(data2plot[subi, :, baselinePeriod1 : baselinePeriod2])
		data2plot_corr[subi, :, :] = data2plot[subi, :, :] - tmp
	data2plot = data2plot_corr


#Determine whether to smooth data
if smooth:
	for subi in np.arange(np.shape(data2plot)[0]):
		print(subi)
		for bini in np.arange(np.shape(data2plot)[1]):
			data2plot[subi, bini, :] = my_smooth(data2plot[subi, bini, :], smoothWindow)

#Initialize matrix 
histos = np.zeros((np.shape(data2plot)[0], len(ListTois), len(n_bins)-1))

#Compute histograms
for subi in np.arange(np.shape(data2plot)[0]):
	for timei in np.arange(len(ListTois)):
		print(subi, timei)
		ind1 = np.abs(times-ListTois[timei][0]).argmin()
		ind2 = np.abs(times-ListTois[timei][1]).argmin()
		histos[subi, timei, :] = np.mean(data2plot[subi, :, ind1 : ind2], axis=1)

#Determine wheter to interpolate histograms
if interpol:
	#n_bins_smooth = np.linspace(-np.pi, np.pi, 500) #resample to more points in order to smooth curve
	histos_spline = np.zeros((np.shape(data2plot)[0], len(ListTois),24))
	#for subi in np.arange(np.shape(data2plot)[0]):
		#for timei in np.arange(len(ListTois)):
			#histos_spline[subi, timei, :] = interpolate.spline(n_bins, histos[subi, timei, :], n_bins_smooth)
	sigma=2
	for subi in np.arange(np.shape(data2plot)[0]):
		for timei in np.arange(len(ListTois)):
			histos_spline[subi, timei, :] = ndimage.gaussian_filter1d(histos[subi, timei, :], sigma)

#Plot 
plt.hold(True)
fig, ax = plt.subplots(1, 1, figsize=[6, 4])

for disti in np.arange(len(ListTois)):
	#plt.plot(np.arange(len(n_bins)-1), np.mean(histos, axis=0)[disti], color = my_colors[disti])
	plt.plot(my_xTicks, np.mean(histos, axis=0)[disti], color = my_colors[disti], linewidth=4)

#Prettify
ax.axvline(0.0, color='gray', linewidth=2, linestyle='dotted') #indicates target position
ax.axvline(-120, color='gray', linewidth=2, linestyle='dotted') #position for counter-clockwise rotation
ax.axvline(120, color='gray', linewidth=2, linestyle='dotted') #position for clockwise rotation
ax.axhline(1./24, color='gray', linewidth=2, linestyle='dotted') #chance

ax.set_xlim(-180, 180)
#ax.set_ylim(0.035, 0.05)

pretty_plot(ax)


fig, ax = plt.subplots(1, 1, figsize=[6, 4])

for disti in np.arange(len(ListTois)):
	#plt.plot(np.arange(len(n_bins)-1), np.mean(histos, axis=0)[disti], color = my_colors[disti])
	plt.plot(my_xTicks, np.mean(histos_spline, axis=0)[disti], color = my_colors[disti], linewidth=4)

#Prettify
ax.axvline(0.0, color='gray', linewidth=2, linestyle='dotted') #indicates target position
ax.axvline(-120, color='gray', linewidth=2, linestyle='dotted') #position for counter-clockwise rotation
ax.axvline(120, color='gray', linewidth=2, linestyle='dotted') #position for clockwise rotation
ax.axhline(1./24, color='gray', linewidth=2, linestyle='dotted') #chance

ax.set_xlim(-180, 180)
#ax.set_ylim(0.035, 0.05)

pretty_plot(ax)
plt.show()