#Purpose: This script computes the distributions of predicted angles from the regression analysis.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 30 March 2018

import sys

import pickle

import numpy as np
import scipy.io as sio

###Define important variables###
ListAnalysis = ['Loc_TrainRot_TestRot']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
#ListSubjects = ['lm130479', 'am150105']
ListSelection = 'RightSeen' #seen, unseen, etc ...
split = 'two-sided'

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'

n_bins = np.linspace(-np.pi, np.pi, 25) #43
time = np.linspace(-0.2, 3.496, 463)
#time = np.linspace(-0.008, 3.392, 18)  #if set to 200ms
#time = np.linspace(-0.104, 3.432, 35)  #if set to 100ms

all_hists = list()

#Load all data
for anali, analysis in enumerate(ListAnalysis):
	for subi, subject in enumerate(ListSubjects):

		print('Loading subject: ' + subject)

		#Load actual predictions
		filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-y_pred.p'
		y_pred = pickle.load(open(filename, 'rb')) #shape: n_train_time, n_test_time, n_trials, labels (predicted angle, radius)
		y_pred = y_pred[:, :, :, 0] #select only the label we are interested in, aka predicted angle

		#Load actual selection of trials to be used
		if ListSelection is 'all':
			sel = np.ones(y_pred.shape[2])
			sel = np.ravel(np.where(sel == 1))
		elif ListSelection is 'seen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelSeen.npy'
			sel = np.load(filename)
		elif ListSelection is 'unseen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelUnseen.npy'
			sel = np.load(filename)
		elif ListSelection is 'RotSeen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelRotSeen.npy'
			sel = np.load(filename)
		elif ListSelection is 'RotUnseen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelRotUnseen.npy'
			sel = np.load(filename)
		elif ListSelection is 'NoRotSeen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelNoRotSeen.npy'
			sel = np.load(filename)
		elif ListSelection is 'NoRotUnseen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelNoRotUnseen.npy'
			sel = np.load(filename)
		elif ListSelection is 'LeftSeen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelLeftSeen.npy'
			sel = np.load(filename)
		elif ListSelection is 'RightSeen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelRightSeen.npy'
			sel = np.load(filename)
		elif ListSelection is 'NoRotSeenCorr':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelNoRotSeenCorr.npy'
			sel = np.load(filename)
		elif ListSelection is 'LeftUnseen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelLeftUnseen.npy'
			sel = np.load(filename)
		elif ListSelection is 'RightUnseen':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelRightUnseen.npy'
			sel = np.load(filename)
		elif ListSelection is 'RightUnseenCorr':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelRightUnseenCorr.npy'
			sel = np.load(filename)
		elif ListSelection is 'LeftUnseenCorr':
			filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-labelLeftUnseenCorr.npy'
			sel = np.load(filename)

		#Load associated labels
		filename = path + analysis + '/IndRes/' + subject + '_Train_Rot_Test_Rot-gatForWeights.p'
		gat = pickle.load(open(filename, 'rb'))
		y_train = gat.y_train_

		#Select only that subset of trials that we are interested in
		y_pred = y_pred[:, :, sel]
		y_train = y_train[sel]

		del gat

		#Sanity check
		if np.shape(y_train)[0] != np.shape(y_pred)[2]:
			print('ERROR: Actual locations and predicted locations do not match!')

		#Compute actual error
		angle_error = np.zeros_like(y_pred)
		angle_error_diag = np.zeros((np.shape(angle_error)[2], len(time)))
		
		for traini in  np.arange(0, len(time)):
			for testi in np.arange(0,  len(time)):
				if split is 'two-sided':
					angle_error[traini, testi, :] = ((y_train - y_pred[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi
					n_bins = np.linspace(-np.pi, np.pi, 25) #43
				else:
					angle_error[traini, testi, :] = np.abs(((y_train - y_pred[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi)
					n_bins = np.linspace(0, np.pi, 13) #43

		#Extract diagonal
		for triali in np.arange(0, np.shape(angle_error)[2]):
			angle_error_diag[triali, :] = np.diag(angle_error[:, :, triali])

		#Compute histogram
		hist = np.zeros((len(n_bins)-1, np.shape(angle_error)[1]))
		for timei in np.arange(0, np.shape(angle_error)[1]):
			tmp = np.histogram(angle_error_diag[:, timei], n_bins)
			hist[:, timei] = tmp[0] / float(np.shape(angle_error)[2])

		all_hists.append(hist)

	#Save data
	np.save(res_path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy', all_hists)

