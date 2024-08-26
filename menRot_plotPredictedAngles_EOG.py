#Purpose: This script plots the distributions of predicted angles from the regression analysis.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 30 March 2018

import sys

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from menRot_plotMat import prettyMat

###Define important variables###
ListAnalysis = ['Loc_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
#ListSubjects = ['lm130479', 'am150105']

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/EOG/'

n_bins = np.linspace(-np.pi, np.pi, 25)
time = np.linspace(-0.2, 3.496, 463)

all_hists = list()

#Load all data
for anali, analysis in enumerate(ListAnalysis):
	for subi, subject in enumerate(ListSubjects):

		#Load actual predictions
		filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-y_pred.p'
		y_pred = pickle.load(open(filename, 'rb')) #shape: n_train_time, n_test_time, n_trials, labels (predicted angle, radius)
		y_pred = y_pred[:, :, :, 0] #select only the label we are interested in, aka predicted angle

		#Load associated labels
		filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-gatForWeights.p'
		gat = pickle.load(open(filename, 'rb'))
		y_train = gat.y_train_

		del gat

		#Sanity check
		if np.shape(y_train)[0] != np.shape(y_pred)[2]:
			print('ERROR: Actual locations and predicted locations do not match!')

		#Compute actual error
		angle_error = np.zeros_like(y_pred)
		angle_error_diag = np.zeros((np.shape(angle_error)[2], 463))
		
		for traini in  np.arange(0, 463):
			for testi in np.arange(0, 463):
				angle_error[traini, testi, :] = ((y_train - y_pred[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi

		#Extract diagonal
		for triali in np.arange(0, np.shape(angle_error)[2]):
			angle_error_diag[triali, :] = np.diag(angle_error[:, :, triali])

		#Compute histogram
		hist = np.zeros((len(n_bins)-1, np.shape(angle_error)[1]))
		for timei in np.arange(0, np.shape(angle_error)[1]):
			tmp = np.histogram(angle_error_diag[:, timei], n_bins)
			hist[:, timei] = tmp[0] / float(np.shape(angle_error)[2])

		all_hists.append(hist)

#Plot data
#im = plt.matshow(hist)
#plt.colorbar(im)
#plt.show()

#im = plt.matshow(hist, origin='lower', aspect='equal')

#prettyMat(all_hists[0], times=time, my_yticks=n_bins, ax=None, cmap='coolwarm')
