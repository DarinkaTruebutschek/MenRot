#Purpose: This script computes the distributions of the classifier estimates as a function of rotation
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 24 May 2018

import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from pycircstat import mean as circMean
from pycircstat import median as circMedian
from pycircstat import watson_williams

from menRot_smooth import my_smooth
from plotBox import plotBox

###Define important variables###
ListAnalysis = ['Resp_TrainAll_TestAll']
vis = 'seen'
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.768], [1.768, 3.268], [3.268, 3.5]]
my_method = 'circMean' #or mean

n_bins = np.linspace(-np.pi, np.pi, 25)
times = np.linspace(-0.2, 3.496, 463)

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Figures/'

#Load data
if vis is 'seen':
	left = np.load(path + 'LeftSeen' + '_' + str(len(n_bins)) + '-histogram.npy')
	noRot = np.load(path + 'NoRotSeen' + '_' + str(len(n_bins)) + '-histogram.npy')
	right = np.load(path + 'RightSeen' + '_' + str(len(n_bins)) + '-histogram.npy')

#Calculate circMean
value_of_interest_left = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))
value_of_interest_noRot = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))
value_of_interest_right = np.zeros((np.shape(data2plot)[0], np.shape(data2plot)[2]))
for subi in np.arange(np.shape(data2plot)[0]):
	#if smooth:
		#for bini in np.arange(np.shape(data2plot)[1]):
			#data2plot[subi, bini, :] = my_smooth(data2plot[subi, bini, :], smoothWindow)
	for linei in np.arange(len(times)):
		if my_method is 'circMean':
			weights = np.linspace((-np.pi + n_bins[1])/2, (np.pi + n_bins[-2])/2, len(n_bins)-1)
			tmp_left = circMean(weights, left[subi, :, linei], ci=None, d=weights[1]-weights[0])
			tmp_noRot = circMean(weights, noRot[subi, :, linei], ci=None, d=weights[1]-weights[0])
			tmp_right = circMean(weights, right[subi, :, linei], ci=None, d=weights[1]-weights[0])
			if tmp_left > np.pi:
				value_of_interest_left[subi, linei] = tmp_left - (2*np.pi)
			else:
				value_of_interest_left[subi, linei] = tmp_left

			if tmp_noRot > np.pi:
				value_of_interest_noRot[subi, linei] = tmp_noRot - (2*np.pi)
			else:
				value_of_interest_noRot[subi, linei] = tmp_noRot

			if tmp_right > np.pi:
				value_of_interest_right[subi, linei] = tmp_right - (2*np.pi)
			else:
				value_of_interest_right[subi, linei] = tmp_right