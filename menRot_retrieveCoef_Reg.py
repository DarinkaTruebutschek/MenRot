#Purpose: Extrace coefficient weights used by classifier and save them as a matlab file.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 22 February 2018

###
import sys
import pickle
import numpy as np

from os import path
from scipy.io import savemat

import jr.gat.scorers 

###Define important variables
dat_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Data/weights/location/'

ListAnalysis = ['Resp_TrainAll_TestAll']
#ListSelection = 'all'

ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

###Extract relevant info
for subi, subject in enumerate(ListSubjects):

	print(subject)

	#Load actual selection of trials to be used
	# if ListSelection is 'all':
	# 	sel = np.ones(y_pred.shape[2])
	# 	sel = np.ravel(np.where(sel == 1))
	# elif ListSelection is 'seen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelSeen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'unseen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelUnseen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'RotSeen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRotSeen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'RotUnseen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRotUnseen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'NoRotSeen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotSeen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'NoRotUnseen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotUnseen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'LeftSeen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftSeen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'RightSeen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRightSeen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'NoRotSeenCorr':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelNoRotSeenCorr.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'LeftUnseen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'RightUnseen':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseen.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'RightUnseenCorr':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseenCorr.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'LeftUnseenCorr':
	# 	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseenCorr.npy'
	# 	sel = np.load(filename)
	# elif ListSelection is 'BothDirSeen':
	# 	filename1 = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftSeen.npy'
	# 	sel1 = np.load(filename1)
	# 	filename2 = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRightSeen.npy'
	# 	sel2 = np.load(filename2)
	# elif ListSelection is 'BothDirUnseen':
	# 	filename1 = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelLeftUnseen.npy'
	# 	sel1 = np.load(filename1)
	# 	filename2 = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-labelRightUnseen.npy'
	# 	sel2 = np.load(filename2)

	#Load associated labels
	filename = dat_path + ListAnalysis[0] + '/IndRes/' + subject + '_Train_All_Test_All-gatForWeights.p'
	gat = pickle.load(open(filename, 'rb'))

	n_class = 1 #should be labels - 1, since this will just return those channels that contributed the most
	n_time = 463
	n_chan = 306
	n_fold = 5

	coef_cos = np.zeros((n_chan,n_time,n_class,n_fold))
	coef_sin = np.zeros((n_chan,n_time,n_class,n_fold))

	for i_time in range(n_time):
		for i_class in range(n_class):
			for i_fold in range(n_fold):
				coef_cos[:, i_time, i_class, i_fold] = gat.estimators_[i_time][i_fold].named_steps['polarregression'].clf_cos.coef_
				#coef_cos[:, i_time, i_class, i_fold] = gat.estimators_[i_time][i_fold].named_steps['polarregression'].clf_cos.coef_[i_class]
				
	savename_cos = res_path + ListSubjects[subi] +'_' + ListAnalysis[0] + '_cos-weights.mat'
	savemat(savename_cos, dict(coef=coef_cos))

	for i_time in range(n_time):
		for i_class in range(n_class):
			for i_fold in range(n_fold):
				coef_sin[:, i_time, i_class, i_fold] = gat.estimators_[i_time][i_fold].named_steps['polarregression'].clf_sin.coef_
				#coef_sin[:, i_time, i_class, i_fold] = gat.estimators_[i_time][i_fold].named_steps['polarregression'].clf_sin.coef_[i_class]

	savename_sin = res_path + ListSubjects[subi] +'_' + ListAnalysis[0] + '_sin-weights.mat'
	savemat(savename_sin, dict(coef=coef_sin))