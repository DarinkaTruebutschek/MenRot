#Purpose: This script computes the slopes of the decoding distributions.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 17 May 2018

import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from numpy.polynomial import polynomial
from scipy.stats import linregress

from pycircstat import mean as circMean
from pycircstat import median as circMedian

from menRot_smooth import my_smooth
from plotBox import plotBox

###Define important variables###
ListAnalysis = ['Loc_TrainAll_TestAll']
ListSelection ='LeftSeen' #'BothDirUnseen' #seen, unseen, etc ...
#ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.768], [1.768, 3.268], [3.268, 3.5]]
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.768], [1.768, 2.5], [2.5, 3.268], [3.268, 3.5]]
my_method = 'circMean' #or mean

n_bins = np.linspace(-np.pi, np.pi, 25)
times = np.linspace(-0.2, 3.496, 463)
#preCue_bin = [100, 246]
#postCue_bin = [246, 433]

smooth = False
smoothWindow = 12#12
smooth2 = False
smoothWindow2 = 1

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Figures/'

if (ListSelection is 'LeftSeen') or (ListSelection is 'RightSeen') or (ListSelection is 'NoRotSeen'):
	tupIndex = np.array([.64, .08, .18])
	ymax = None
	ymin = None
elif (ListSelection is 'LeftUnseen') or (ListSelection is 'RightUnseen') or (ListSelection is 'NoRotUnseen'):
	tupIndex = np.array([.2, .3, .49])
	ymax = None
	ymin = None
elif vis is 'unseenCorr':
	tupIndex = np.array([0, .45, .74])
	ymax = 1.2
	ymin = -.6
elif vis is 'unseenIncorr':
	tupIndex = np.array([.39, .47, .64])
	ymax = 1.2
	ymin = -.6
elif vis is 'RotSeen':
	tupIndex = np.array([.64, .08, .18])
	ymin = -.75
	ymax = 1.0


#Load data
data2plot = np.load(path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy') #subjects x n_bins x time

#Determine whether or not to smooth the data
if smooth:
	for subi in np.arange(np.shape(data2plot)[1]):
		for bini in np.arange(np.shape(data2plot)[1]):
			data2plot[subi, bini, :] = my_smooth(data2plot[subi, bini, :], smoothWindow)

#Find only that part of the distribution that exceeds chance performance
#belChance = data2plot < (1./(len(n_bins)-1))
#data2plot[belChance] = 0

#First extract for each subject and time point
value_of_interest = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))

#for subi in np.arange(1):
for subi in np.arange(np.shape(data2plot)[0]):
	#if smooth:
		#for bini in np.arange(np.shape(data2plot)[1]):
			#data2plot[subi, bini, :] = my_smooth(data2plot[subi, bini, :], smoothWindow)
	for linei in np.arange(len(times)):
		if my_method is 'max':
			tmp = np.abs(data2plot[subi, :, linei] - np.max(data2plot[subi, :, linei])).argmin()
			value_of_interest[subi, linei] = n_bins[tmp]
		elif my_method is 'maxThresh':
			tmp = np.abs(data2plot[subi, :, linei] - np.median(data2plot[subi, :, linei] > (1./len(n_bins)-1))).argmin() #only consider that part of the distribution that is above chance
			value_of_interest[subi, linei] = n_bins[tmp]
		elif my_method is 'median':
			tmp = np.abs(data2plot[subi, :, linei] - np.median(data2plot[subi, :, linei])).argmin()
			value_of_interest[subi, linei] = n_bins[tmp]
		elif my_method is 'medianThresh':
			tmp = np.abs(data2plot[subi, :, linei] - np.median(data2plot[subi, :, linei] > (1./len(n_bins)-1))).argmin() #only consider that part of the distribution that is above chance
			value_of_interest[subi, linei] = n_bins[tmp]
		elif my_method is 'expectedValue':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			tmp = np.sum(data2plot[subi, :, linei] * weights)
			value_of_interest[subi, linei] = tmp
		elif my_method is 'thirdMoment':
			weights = (np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1))**3
			tmp = np.cbrt(np.sum(data2plot[subi, :, linei] * weights))
			value_of_interest[subi, linei] = tmp
		elif my_method is 'fifthMoment':
			weights = (np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1))**5
			tmp = (np.sum(data2plot[subi, :, linei] * weights))
			if np.isnan(tmp) is False:
				tmp = tmp ** (1.0/5)
			else:
				np.isnan(tmp)
			value_of_interest[subi, linei] = np.round(tmp, 5)
		elif my_method is 'circMean':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			tmp = circMean(weights, data2plot[subi, :, linei], ci=None, d=weights[1]-weights[0])
			if tmp > np.pi:
				value_of_interest[subi, linei] = tmp - (2*np.pi)
			else:
				value_of_interest[subi, linei] = tmp 
	if smooth2:
		value_of_interest[subi, :] = my_smooth(value_of_interest[subi, :], smoothWindow2)


#Next, fit straight line seperately for pre-cue and post-cue period
intercept = np.zeros((np.shape(data2plot)[0], len(ListTois)))
slope = np.zeros((np.shape(data2plot)[0], len(ListTois)))

#for subi in np.arange(1):
for subi in np.arange(np.shape(data2plot)[0]):
	for t, toi in enumerate(ListTois):
		timei1 = np.abs(times - toi[0]).argmin()
		timei2 = np.abs(times - toi[1]).argmin()
		#fit = polynomial.polyfit(times[preCue_bin[0] : preCue_bin[1]], value_of_interest[subi, preCue_bin[0] : preCue_bin[1]], 1) 
		fit = linregress(times[timei1 : timei2], value_of_interest[subi, timei1 : timei2]) 
		slope[subi, t] = fit[0]
		intercept[subi, t] = fit[1]

		#sns.regplot(times[timei1 : timei2], value_of_interest[subi, timei1 : timei2])
		#plt.savefig(res_path + 'Regression' + ListSelection + '_' + my_method + '_' + str(smoothWindow) + '_' + str(len(n_bins)) + '_' + str(timei1) + '.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
		#plt.show()

#Compute significance
p_values = np.zeros((len(ListTois)))
for t in np.arange(len(ListTois)):
	p_values[t] = scipy.stats.wilcoxon(slope[:, t])[1]
	#if p_values[t] < (.05/(len(ListTois))):
		#p_values[t] = p_values[t]
	#else:
		#p_values[t] = 1
#Plot
ax = plotBox(slope, scipy.stats.sem(slope, axis=0), p_values, tupIndex, ymax, ymin)

#plt.savefig(res_path + 'Boxplot_Slopes_' + ListSelection + '_' + my_method + '_'  + '_' str(len(nbins)) + '.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
plt.savefig(res_path + 'Boxplot_Slopes' + ListSelection + '_' + my_method + '_' + str(smoothWindow) + '_' + str(len(n_bins)) + '.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
plt.show()
#plt.savefig(res_path + 'Boxplot_Slopes_' + ListSelection + '_' + my_method + '_' + str(smoothWindow) + '_' str(len(nbins)) + '.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
