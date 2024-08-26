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
dat_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Data/weights/'

ListAnalysis = ['Train_Rot_Test_Rot']

ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

###Extract relevant info
for subi in range(len(ListSubjects)):
	filename = dat_path + ListAnalysis[0] + '/IndRes/' + ListSubjects[subi] + '_' + ListAnalysis[0] + '-gatForWeights.p'
	gat = pickle.load(open(filename, 'rb'))

	n_class = 1 #should be labels - 1, since this will just return those channels that contributed the most
	n_time = 463
	n_chan = 306
	n_fold = 5

	coef = np.zeros((n_chan,n_time,n_class,n_fold))

	for i_time in range(n_time):
		for i_class in range(n_class):
			for i_fold in range(n_fold):
				coef[:,i_time,i_class,i_fold] = gat.estimators_[i_time][i_fold].named_steps['svc'].coef_[i_class]


	savename = res_path + ListSubjects[subi] +'_' + ListAnalysis[0] + 'weights.mat'
	savemat(savename, dict(coef=coef))