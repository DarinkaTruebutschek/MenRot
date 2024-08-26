#Purpose: This script computes the distributions of predicted angles from the regression analysis.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 30 March 2018

import sys

import pickle

import numpy as np
import scipy.io as sio

from menRot_smooth import my_smooth

def computePredictedAngles(my_analysis, my_visibility, my_bins, my_split):

	###Define important variables###
	ListAnalysis = [my_analysis]	
	ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
		'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
		'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
		'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
		'cc130066', 'in110286', 'ss120102']
	ListSelection = my_visibility #seen, unseen, etc ...
	split = my_split
	smooth = False
	smoothWindow = 2

	if ListAnalysis[0] is not 'InferResp_TrainAll_TestAll':
		path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/'
		res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
	else:
		path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/'
		res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/' + ListAnalysis[0] + '/GroupRes/Stats/'

	my_bins = my_bins
	time = np.linspace(-0.2, 3.496, 463)

	all_hists = list()

	#Load all data
	for anali, analysis in enumerate(ListAnalysis):
		for subi, subject in enumerate(ListSubjects):

			print('Loading subject: ' + subject)

			#Load actual predictions
			filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-y_pred.p'
			y_pred = pickle.load(open(filename, 'rb')) #shape: n_train_time, n_test_time, n_trials, labels (predicted angle, radius)
			y_pred = y_pred[:, :, :, 0] #select only the label we are interested in, aka predicted angle

			#Load actual selection of trials to be used
			if ListSelection is 'all':
				sel = np.ones(y_pred.shape[2])
				sel = np.ravel(np.where(sel == 1))
			elif ListSelection is 'seen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelSeen.npy'
				sel = np.load(filename)
			elif ListSelection is 'unseen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelUnseen.npy'
				sel = np.load(filename)
			elif ListSelection is 'RotSeen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRotSeen.npy'
				sel = np.load(filename)
			elif ListSelection is 'RotUnseen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRotUnseen.npy'
				sel = np.load(filename)
			elif ListSelection is 'NoRotSeen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotSeen.npy'
				sel = np.load(filename)
			elif ListSelection is 'NoRotUnseen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotUnseen.npy'
				sel = np.load(filename)
			elif ListSelection is 'NoRotUnseenCorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotUnseenCorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'NoRotUnseenIncorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotUnseenIncorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'LeftSeen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftSeen.npy'
				sel = np.load(filename)
			elif ListSelection is 'RightSeen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightSeen.npy'
				sel = np.load(filename)
			elif ListSelection is 'NoRotSeenCorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotSeenCorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'LeftUnseen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseen.npy'
				sel = np.load(filename)
			elif ListSelection is 'RightUnseen':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseen.npy'
				sel = np.load(filename)
			elif ListSelection is 'RightUnseenCorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseenCorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'LeftUnseenCorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseenCorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'RightUnseenIncorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseenIncorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'LeftUnseenIncorr':
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseenIncorr.npy'
				sel = np.load(filename)
			elif ListSelection is 'BothDirSeen':
				filename1 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftSeen.npy'
				sel1 = np.load(filename1)
				filename2 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightSeen.npy'
				sel2 = np.load(filename2)
			elif ListSelection is 'BothDirUnseen':
				filename1 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseen.npy'
				sel1 = np.load(filename1)
				filename2 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseen.npy'
				sel2 = np.load(filename2)
			elif ListSelection is 'BothDirUnseenCorr':
				filename1 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseenCorr.npy'
				sel1 = np.load(filename1)
				filename2 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseenCorr.npy'
				sel2 = np.load(filename2)
			elif ListSelection is 'BothDirUnseenIncorr':
				filename1 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseenIncorr.npy'
				sel1 = np.load(filename1)
				filename2 = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseenIncorr.npy'
				sel2 = np.load(filename2)
			#Load associated labels
			if (analysis is not 'Resp_TrainAll_TestAll') & (analysis is not 'Resp_TrainRot_TestRot'):
				filename = path + analysis + '/IndRes/' + subject + '_Train_All_Test_All-gatForWeights.p'
			else:
				filename = path + 'Loc_TrainAll_TestAll/IndRes/' + subject + '_Train_All_Test_All-gatForWeights.p' #if estimating angle for resp, we still want to compare it to actual target loc
				#filename = path + 'Infer_TrainAll_TestAll/IndRes/' + subject + '_Train_All_Test_All-gatForWeights.p' #if estimating angle for resp, we still want to compare it to actual target loc
			gat = pickle.load(open(filename, 'rb'))
			y_train = gat.y_train_

			#Select only that subset of trials that we are interested in
			if (ListSelection is not 'BothDirSeen') and (ListSelection is not 'BothDirUnseen') and (ListSelection is not 'BothDirUnseenCorr') and (ListSelection is not 'BothDirUnseenIncorr'):
				y_pred = y_pred[:, :, sel]
				y_train = y_train[sel]
			else:
				y_pred1 = y_pred[:, :, sel1]
				y_train1 = y_train[sel1]
				y_pred2 = y_pred[:, :, sel2]
				y_train2 = y_train[sel2]

			del gat

			#Sanity check
			if np.shape(y_train)[0] != np.shape(y_pred)[2]:
				print('ERROR: Actual locations and predicted locations do not match!')

			#Compute actual error
			if (ListSelection is not 'BothDirSeen') and (ListSelection is not 'BothDirUnseen') and (ListSelection is not 'BothDirUnseenCorr') and (ListSelection is not 'BothDirUnseenIncorr'):
				angle_error = np.zeros_like(y_pred)
				angle_error_diag = np.zeros((np.shape(angle_error)[2], len(time)))
		
				for traini in  np.arange(0, len(time)):
					for testi in np.arange(0,  len(time)):
						if split is 'two':
							angle_error[traini, testi, :] = ((y_train - y_pred[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi
							n_bins = my_bins #43
						else:
							angle_error[traini, testi, :] = np.abs(((y_train - y_pred[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi)
							n_bins = my_bins #43

				#Extract diagonal
				for triali in np.arange(0, np.shape(angle_error)[2]):
					angle_error_diag[triali, :] = np.diag(angle_error[:, :, triali])
			else:
				angle_error1 = np.zeros_like(y_pred1)
				angle_error2 = np.zeros_like(y_pred2)

				angle_error_diag = np.zeros(((np.shape(angle_error1)[2]) + (np.shape(angle_error2)[2]), len(time)))
		
				for traini in  np.arange(0, len(time)):
					for testi in np.arange(0,  len(time)):
						if split is 'two':
							angle_error1[traini, testi, :] = ((y_train1 - y_pred1[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi
							n_bins = my_bins #43

							angle_error2[traini, testi, :] = ((y_train2 - y_pred2[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi
							n_bins = my_bins #43
						else:
							angle_error1[traini, testi, :] = np.abs(((y_train1 - y_pred1[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi)
							n_bins = my_bins #25

							angle_error2[traini, testi, :] = np.abs(((y_train2 - y_pred2[traini, testi, :]) + np.pi) % (2 * np.pi) - np.pi)
							n_bins = my_bins #25

				#Renorm for left-sided rotations
				tmp1 = angle_error1 < 0
				tmp2 = angle_error1 > 0

				angle_error1[tmp1] = np.abs(angle_error1[tmp1])
				angle_error1[tmp2] = -(angle_error1[tmp2])

				#Collapse the two directions into a single matrix
				angle_error = np.concatenate((angle_error1, angle_error2), axis=2)
			
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
		if smooth:
			#np.save(res_path + ListSelection + '_' + str(len(n_bins)) + 'withRespectInfer-smoothed-histogram.npy', all_hists)
			np.save(res_path + ListSelection + '_' + str(len(n_bins)) + '-smoothed-histogram.npy', all_hists)
		else:
			#np.save(res_path + ListSelection + '_' + str(len(n_bins)) + 'withRespectInfer-histogram.npy', all_hists)
			np.save(res_path + ListSelection + '_' + str(len(n_bins)) + '-histogram.npy', all_hists)

