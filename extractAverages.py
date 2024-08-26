#Purpose: Extract mean/sd for given time bin and compute significance.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 12 March 2018

###Setup
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio

from scipy import stats

def extractAverages(data, time, toi, chance, test):

	#Check whether time and data have same dimension
	if np.shape(data[1]) != np.shape(time):
		print('Error: Time dimension and dimension of data do not match!')

	#Find indices for given timeBin
	ind1 = np.where(time == toi[0])
	ind1 = int(ind1[0][0])

	ind2 = np.where(time == toi[1])
	ind2 = int(ind2[0][0])

	#Compute mean and standard error	
	dat_mean = np.mean(data[:, ind1 : ind2], axis = 1)
	dat_sem = stats.sem(dat_mean, axis = 0)

	#Compute p_values
	if test is 'Wilcoxon':
		tmp = stats.wilcoxon(dat_mean - chance)
		dat_pvalues = tmp[1] #only get p-value without test statistic
		dat_stats = tmp[0]
	elif test is 'ttest':
		tmp = stats.ttest_1samp(dat_mean, chance, axis=0) 
		dat_pvalues = tmp[1]
		dat_stats = tmp[0]

	return (dat_mean, dat_sem, dat_pvalues, dat_stats)