#Purpose: This script runs the computation of the distributions of predicted angles from the regression analysis.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 30 March 2018

import sys

import pickle

import numpy as np
import scipy.io as sio

from menRot_computePredictedAngles_slice import computePredictedAngles_slice

###Define important variables###
#ListAnalysis = ['Resp_TrainAll_TestAll', 'Infer_TrainAll_TestAll', 'Loc_TrainAll_TestAll']
ListAnalysis = ['Infer_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

my_slice = [0.5, 0.508]
my_split = 'two'
my_bins = np.linspace(-np.pi, np.pi, 13) #43
#my_bins = np.linspace(-np.pi, np.pi, 7) #43
smooth = False
smoothWindow = 2

for analysis_i, my_analysis in enumerate(ListAnalysis):
	print (my_analysis)

	#ListSelection = ['BothDirSeen', 'BothDirUnseen', 'NoRotSeen', 'NoRotUnseen', 'LeftSeen', 'LeftUnseen', 'RightSeen', 'RightUnseen'] #seen, unseen, etc ...
	ListSelection = ['BothDirSeen'] #seen, unseen, etc ...

	for visi, visibility in enumerate(ListSelection):
		print(visibility)

		computePredictedAngles_slice(my_analysis, visibility, my_bins, my_split, my_slice)
