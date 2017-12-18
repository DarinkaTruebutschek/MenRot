#Purpose: Compute between-subject statistics for decoding analyses.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 18 December 2017

import sys

import numpy as np
import scipy.io as sio

from menRot_wilcoxon import _my_wilcoxon
from scipy import stats

###Define important variables###
ListAnalysis = ['Loc_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105']
beginTime = -0.2

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding'

###Load decoding results (GAT, time)
for analysis in ListAnalysis:

	dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
	res_path = path + '/' + ListAnalysis[0] + '/GroupRes'

	print('load: ' + analysis)
	all_scores = list() #initialize matrix containing all of GATs

	for subject in ListSubjects:
		score = np.load(dat_path + '/' + subject + '_Train_All_Test_All-score.npy') #load actual data
		time = np.load(dat_path + '/' + subject + '_Train_All_Test_All-time.npy') #load timing info
		all_scores.append(score) #matrix: NSubs x NTrain x NTest

		#Check whether time and data have the same axis
		if len(score[0]) != len(time):
			print('Updating and saving time axis now ...')

			tmp = np.where(time == beginTime)
			tmp = tmp[0][0]

			time = time[tmp :]

			np.save(res_path + '/' + subject + '_Train_All_Test_All-time.npy', time)

	








