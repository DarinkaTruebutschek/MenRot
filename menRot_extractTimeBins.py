#Purpose: This script computes the average decoding performance over specific time windows and plots the results.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 22 February 2018

###
import sys
import numpy as np

### Define important variables
dat_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis/'

ListAnalysis = 'Train_All_Test_All'
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.768], [1.768, 3.268], [3.268, 3.5]]

ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

beginTime = -0.2
endTime = 3.496

### Initialize variables
all_scores = np.zeros((len(ListTois), len(ListSubjects)))

### Load data
time = np.load(dat_path + ListAnalysis + '/GroupRes/Stats/' + ListAnalysis + '-time.npy') #load timing

for timei, toi in enumerate(ListTois):
	print(toi)

	for subi, subject in enumerate(ListSubjects):
		print(subject)

		score = np.load(dat_path + ListAnalysis + '/IndRes/' + subject + '_' + ListAnalysis + '-score.npy')


