#Purpose: Compute between-subject statistics for decoding analyses (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 18 December 2017

import sys

import numpy as np
import scipy.io as sio

from menRot_base import myStats, parallel_stats
from scipy import stats

###Define important variables###
ListAnalysis = ['Loc_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338']

beginTime = -0.2
endTime = 3.496

chance = 0 #for analyses involving 

stat_params = 'permutation'


#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding'

###Load decoding results (GAT, time)
for analysis in ListAnalysis:

	dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
	res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'

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

			np.save(res_path + '/' + subject + analysis + '-time.npy', time)

	if stat_params is 'permutation':
		#Compute stats against theoretical chance level as obtained by permutations
		print('computing stats based on permutation: ' + analysis)

		p_values = myStats(np.array(all_scores) - chance) #p_values for entire gat (same shape as np.mean(gat))

		diag_offdiag = all_scores - np.tile([np.diag(sc) for sc in all_scores], [len(time), 1, 1]).transpose(1, 0, 2)
		p_values_off = myStats(diag_offdiag)

		scores_diag = [np.diag(sc) for sc in all_scores]
		p_values_diag = myStats(np.array(scores_diag)[:, :, None] - chance)

	elif stat_params is 'Wilcoxon':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on uncorrected Wilcoxon: ' + analysis)

		p_values = parallel_stats(np.array(all_scores) - chance, correction=False) #p_values for entire gat (same shape as np.mean(gat))

		diag_offdiag = all_scores - np.tile([np.diag(sc) for sc in all_scores], [len(time), 1, 1]).transpose(1, 0, 2)
		p_values_off = parallel_stats(np.array(diag_offdiag) - chance, correction=False) 

		scores_diag = [np.diag(sc) for sc in all_scores]
		p_values_diag = parallel_stats(np.array(scores_diag) - chance, correction=False) 

	elif stat_params is 'Wilcoxon-FDR':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on corrected Wilcoxon: ' + analysis)

		p_values = parallel_stats(np.array(all_scores) - chance, correction='FDR') #p_values for entire gat (same shape as np.mean(gat))

		diag_offdiag = all_scores - np.tile([np.diag(sc) for sc in all_scores], [len(time), 1, 1]).transpose(1, 0, 2)
		p_values_off = parallel_stats(np.array(diag_offdiag) - chance, correction='FDR') 

		scores_diag = [np.diag(sc) for sc in all_scores]
		p_values_diag = parallel_stats(np.array(scores_diag) - chance, correction='FDR') 

	#Save
	np.save(res_path + '/' + analysis + '_' + stat_params + '-p_values.npy', p_values)
	np.save(res_path + '/' + analysis + '_' + stat_params + '-p_values_off.npy', p_values_off)
	np.save(res_path + '/' + analysis + '_' + stat_params + '-p_values_diag.npy', p_values_diag)









