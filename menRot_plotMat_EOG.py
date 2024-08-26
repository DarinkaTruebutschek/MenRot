#Purpose: This script plots the distributions of predicted angles a pretty matrix plot.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 30 March 2018

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.stats as stats
import seaborn as sns

from menRot_base_plot import pretty_plot, plot_sem, plot_widths, pretty_colorbar
from menRot_plotGat import _set_ticks
from menRot_smooth import my_smooth
from matplotlib.colors import ListedColormap

from pycircstat import mean as circMean
from pycircstat import median as circMedian

###Define important variables###
ListAnalysis = ['Resp_TrainAll_TestAll']
ListSelection = 'RightSeen' #'BothDirUnseen' #seen, unseen, etc ...

my_method = 'circMean'
my_method_Group = 'circMean'

smooth =True
smooth2 = True
smoothWindow = 12#4
smoothWindow2 = 12

baselineCorrection=None

stat_params = 'permutation'
tail = 1
sig =None

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/EOG/' + ListAnalysis[0] + '/GroupRes/Stats/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/EOG/' + ListAnalysis[0] + '/GroupRes/Figures/'

n_bins = np.linspace(-np.pi, np.pi, 25) #25
times = np.linspace(-0.2, 3.496, 463)

#Load data
if (ListAnalysis[0] is 'Resp_TrainAll_TestAll'): 
	if ((ListSelection is'RightSeen') or (ListSelection is 'NoRotSeen') or (ListSelection is  'LeftSeen')) or ((ListSelection is'RightUnseen') or (ListSelection is 'NoRotUnseen') or (ListSelection is  'LeftUnseen')) or ((ListSelection is'RightUnseenCorr') or (ListSelection is 'NoRotUnseenCorr') or (ListSelection is  'LeftUnseenCorr')) or ((ListSelection is'RightUnseenIncorr') or (ListSelection is 'NoRotUnseenIncorr') or (ListSelection is  'LeftUnseenIncorr')) or (ListSelection is  'BothDirUnseen') or (ListSelection is  'BothDirUnseenIncorr'):
		#data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + 'withRespectInfer-histogram.npy') #subjects x n_bins x time
		data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy') #subjects x n_bins x time
	else:
		data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy')
else:
	data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy')
if sig is not None:
	stats2plot = np.load(path  + ListAnalysis[0] + '_' + stat_params +  str(tail) +  '_' + str(len(n_bins)) + '-' + ListSelection + 'predictedAngles-p_values.npy')
#data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '_Slices_0.5_to_0.508-histogram.npy') #subjects x n_bins x time
data2plot = np.array(data2plot)

#Determine whether or not to smooth the data
if smooth:
	for subi in np.arange(np.shape(data2plot)[1]):
		for bini in np.arange(np.shape(data2plot)[1]):
			data2plot[subi, bini, :] = my_smooth(data2plot[subi, bini, :], smoothWindow)

#Find only that part of the distribution that exceeds chance performance
#belChance = data2plot < (1./(len(n_bins)-1))
#data2plot[belChance] = 0

#for subi in np.arange(np.shape(data2plot)[1]):
	#for linei in np.arange(len(times)):
		#data2plot[subi, :, linei] = data2plot[subi, :, linei] / (np.sum(data2plot[subi, :, linei], axis=0))

#Compute maximum for each column
line2plot = np.zeros((len(times)))
line2plot1 = np.zeros((len(times)))
line2plot2 = np.zeros((len(times)))
value_of_interest = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))
value_of_interest1 = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))
value_of_interest2 = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))

for subi in np.arange(np.shape(data2plot)[0]):
	#if smooth:
		#for bini in np.arange(np.shape(data2plot)[1]):
			#data2plot[subi, bini, :] = my_smooth(data2plot[subi, bini, :], smoothWindow)
	for linei in np.arange(len(times)):
		if my_method is 'max':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			tmp = np.abs(data2plot[subi, :, linei] - np.max(data2plot[subi, :, linei])).argmin()
			value_of_interest[subi, linei] = weights[tmp]
		elif my_method is 'maxThresh':
			tmp = np.abs(data2plot[subi, :, linei] - np.median(data2plot[subi, :, linei] > (1./len(n_bins)-1))).argmin() #only consider that part of the distribution that is above chance
			value_of_interest[subi, linei] = n_bins[tmp]
		elif my_method is 'median':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			#tmp = np.abs(data2plot[subi, :, linei] - np.median(data2plot[subi, :, linei])).argmin()
			tmp = np.argsort(data2plot[subi, :, linei])
			value_of_interest[subi, linei] = np.mean(weights[tmp[11]] + weights[tmp[12]])
		elif my_method is 'medianThresh':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			#tmp = np.abs(data2plot[subi, :, linei] - np.median(data2plot[subi, :, linei] > (1./len(n_bins)-1))).argmin() #only consider that part of the distribution that is above chance
			#value_of_interest[subi, linei] = weights[tmp]
			#value_of_interest[subi, linei] = np.median(data2plot[subi, :, linei] > (1./len(n_bins)-1))
			tmp = data2plot[subi, :, linei] > (1./len(n_bins)-1) #which values actually exceed chance
			tmp2 = weights[tmp] #those positions that correspond to above-chance performance
			tmp3 = np.argsort(data2plot[subi, :, linei][tmp])
			tmp2 = tmp2[tmp3] #sort according to previous sorting
			if len(tmp2) % 2 == 0: #even number
				value_of_interest[subi, linei] = np.mean(tmp2[(len(tmp2)/2)-1 : len(tmp2)/2])
			else:
				value_of_interest[subi, linei] = tmp2[len(tmp2)//2]
			#value_of_interest[subi, linei] = np.median(data2plot[subi, :, linei] > (1./len(n_bins)-1))
		elif my_method is 'circMean':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			tmp = circMean(weights, data2plot[subi, :, linei], ci=None, d=weights[1]-weights[0])
			if tmp > np.pi:
				value_of_interest[subi, linei] = tmp - (2*np.pi)
			else:
				value_of_interest[subi, linei] = tmp 
		elif my_method is 'expectedValue':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			tmp = np.sum(data2plot[subi, :, linei] * weights)
			value_of_interest[subi, linei] = tmp
		elif my_method is 'thirdMoment':
			weights = (np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1))**3
			tmp = np.cbrt(np.sum(data2plot[subi, :, linei] * weights))
			value_of_interest[subi, linei] = tmp
		elif my_method is 'percentile':
			#tmp1 = np.abs(data2plot[subi, :, linei] - np.percentile((data2plot[subi, :, linei] > (1./len(n_bins)-1)), 50)).argmin() #only consider that part of the distribution that is above chance
			#tmp2 = np.abs(data2plot[subi, :, linei] - np.percentile((data2plot[subi, :, linei] > (1./len(n_bins)-1)), 95)).argmin() #only consider that part of the distribution that is above chance
			tmp1 = np.abs(data2plot[subi, :, linei] - np.percentile(data2plot[subi, :, linei], 1)).argmin()
			tmp2 = np.abs(data2plot[subi, :, linei] - np.percentile(data2plot[subi, :, linei], 99)).argmin()
			value_of_interest1[subi, linei] = n_bins[tmp1]
			value_of_interest2[subi, linei] = n_bins[tmp2]
	if smooth2:
		value_of_interest[subi, :] = my_smooth(value_of_interest[subi, :], smoothWindow2)

if my_method is 'expectedGroup':
	for linei in np.arange(len(times)):
		weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
		tmp = np.sum(np.mean(data2plot, axis=0)[:, linei] * weights)
		line2plot[linei] = tmp
elif my_method is 'medianThreshGroup':
	for linei in np.arange(len(times)):
		tmp = np.abs(np.mean(data2plot, axis=0)[:, linei] - np.median(np.mean(data2plot, axis=0)[:, linei] > (1./len(n_bins)-1))).argmin() #only consider that part of the distribution that is above chance
		line2plot[linei] = n_bins[tmp]
		if smooth2:
			line2plot[linei] = my_smooth(line2plot[linei], smoothWindow2)
elif my_method is 'thirdMomentGroup':
	for linei in np.arange(len(times)):
		weights = (np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1))**3
		tmp = np.cbrt(np.sum(np.mean(data2plot, axis=0)[:, linei] * weights))
		line2plot[linei] = tmp
elif my_method is 'maxGroup':
	for linei in np.arange(len(times)):
		weights = (np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1))
		tmp = np.abs(np.mean(data2plot, axis=0)[:, linei] - np.max(np.mean(data2plot, axis=0)[:, linei])).argmin()
		line2plot[linei] = weights[tmp]
	if smooth2:
		line2plot = my_smooth(line2plot, smoothWindow2)

if my_method_Group is 'standard':
	if my_method is not 'percentile':
		line2plot = np.mean(value_of_interest, axis=0)
	else:
		line2plot1 = np.mean(value_of_interest1, axis=0)
		line2plot2 = np.mean(value_of_interest2, axis=0)
elif my_method_Group is 'circMean':
	if my_method is not 'percentile':
		line2plot = circMean(value_of_interest, axis=0)
		line2plot[line2plot > np.pi] = line2plot[line2plot > np.pi] - (2*np.pi)
elif my_method_Group is 'circMedian':
	if my_method is not 'percentile':
		line2plot = circMedian(value_of_interest, axis=0)
		line2plot[line2plot > np.pi] = line2plot[line2plot > np.pi] - (2*np.pi)

#Prepare plot
extent = [min(times), max(times), min(n_bins), max(n_bins)]

#Plot
plt.hold(True)

if ListAnalysis[0] is 'Resp_TrainAll_TestAll':
	fig, ax = plt.subplots(1, 1, figsize=[4, 4])
else:
	fig, ax = plt.subplots(1, 1, figsize=[8, 4])

if sig is not None:
	sig = np.mean(data2plot, axis=0)
	sig[stats2plot > .1] = 0
	im = ax.matshow(sig, extent=extent, cmap='coolwarm', origin='lower', aspect='auto', vmin=1./24, vmax=0.08)
	im.cmap.set_under('w', alpha=0)
else:
	#current_palette = sns.color_palette('Blues')
	current_palette = sns.color_palette('RdBu_r', 7)
	#current_palette = sns.color_palette('coolwarm', 7)
	#current_palette = sns.diverging_palette(220, 10, sep=80, n=7)
	my_cmap=ListedColormap(sns.color_palette(current_palette).as_hex())
	im = ax.matshow(np.mean(data2plot, axis=0), extent=extent, cmap=my_cmap, origin='lower', aspect='auto', vmin=1./24, vmax=0.06)
	#im.cmap.set_under('w', alpha=0)
	#im = ax.contourf(np.mean(data2plot, axis=0), levels=np.linspace(0.042, 0.075, 10), extent=extent, cmap='coolwarm', origin='lower', aspect='equal')

#im = ax.contourf(np.mean(data2plot, axis=0), levels=np.linspace(0.048, 0.075, 5), extent=extent, cmap='coolwarm', origin='lower', aspect='equal')
#plt.contour(np.mean(data2plot, axis=0), levels=np.linspace(0.048, 0.075, 5), extent=extent, origin='lower', aspect='equal', linewidths = 0.2, colors = 'k')


#Plot line overlaying max
if my_method is not 'percentile':
	plt.plot(times, line2plot, color = 'k', linewidth=2)
	if my_method_Group is not 'circMean':
		plt.fill_between(times, line2plot-stats.sem(value_of_interest, axis=0), line2plot+stats.sem(value_of_interest, axis=0), edgecolor='none', facecolor='dimgray', alpha=.5)
	else:
		plt.fill_between(times, line2plot-(stats.circstd(value_of_interest, high = np.pi, axis=0) / np.sqrt(np.shape(data2plot)[0])), line2plot+(stats.circstd(value_of_interest, high = np.pi, axis=0) / np.sqrt(np.shape(data2plot)[0])), edgecolor='none', facecolor='dimgray', alpha=.5)
else:
	plt.plot(times, line2plot1, color = 'k', linewidth=2)
	plt.fill_between(times, line2plot1-stats.sem(value_of_interest, axis=0), line2plot1+stats.sem(value_of_interest, axis=0), edgecolor='none', facecolor='dimgray', alpha=.5)

	plt.plot(times, line2plot2, color = 'k', linewidth=2)
	plt.fill_between(times, line2plot2-stats.sem(value_of_interest, axis=0), line2plot2+stats.sem(value_of_interest, axis=0), edgecolor='none', facecolor='dimgray', alpha=.5)

#Plot sig
#if sig is not None:
	#sig = stats2plot < .05
	#xx, yy = np.meshgrid(times, np.arange(np.shape(n_bins)[0]-1), copy=False, indexing='xy')
	#ax.contour(xx, yy, sig, colors= 'k', linestyles='solid', linewidths = 0.25)

#Add event markers
ax.axvline(0.0, color='k', linewidth=1) #indicates target onset
ax.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
ax.axvline(3.2667, color='k', linewidth=1) #indicates response onset

ax.axhline(0.0, color='k', linestyle=':', linewidth=2)
ax.axhline(-2.09, color='k', linestyle=':', linewidth=2)
ax.axhline(2.09, color='k', linestyle=':', linewidth=2)

#Setup ticks
ax.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
ax.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 16})

ax.set_yticks(n_bins)
#ax.set_yticklabels([r'$-\pi$', ' ', r'$-\frac{2}{3}\pi$', ' ', ' ', ' ', '0', ' ', ' ', r'$\frac{2}{3}\pi$', ' ', ' ', r'$\pi$'], fontdict={'family': 'arial', 'size': 12})
#ax.set_yticklabels(np.rad2deg(n_bins))
#ax.set_yticklabels(['', '', '', '', '+120', '', '', '', '', '', '', '', 'Pre-   \nrotation', '', '', '', '', '', '', '', '-120', '', '', '', ''], fontdict={'family': 'arial', 'size': 16})
ax.set_yticklabels(['', '', '', '', '+120', '', '', '', '', '', '', '', 'Target', '', '', '', '', '', '', '', '-120', '', '', '', ''], fontdict={'family': 'arial', 'size': 16})

if ListAnalysis[0] is 'Resp_TrainAll_TestAll':
	ax.set_xlim(times[245], max(times))
	#ax.set_ylim(min(n_bins), max(n_bins))
else:
	ax.set_xlim(min(times), max(times))
ax.set_ylim(min(n_bins), max(n_bins))

#pretty_colorbar()

pretty_plot(ax)

plt.savefig(res_path + ListSelection + '_' + str(len(n_bins)) + my_method + '-' + my_method_Group + '-histogram.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
plt.show()

#for subi in np.arange(30):
	#fig, ax = plt.subplots(1, 1, figsize=[8, 3])
	#im = ax.matshow(data2plot[subi, :], extent=extent, cmap='coolwarm', origin='lower', aspect='auto', vmin=1./24)
	#plt.show()


