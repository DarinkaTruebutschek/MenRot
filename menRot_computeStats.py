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
ListAnalysis = ['Resp_Trainloc_TestAllSeen']
ListTois = [[0.096, 0.296], [0.296, 0.6], [0.6, 0.8], [0.296, 0.8], [0.096, 0.8]]
#ListSubjects = ['lm130479', 'am150105', 'cb140229']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

#ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	#'ag150338', 'ml140071', 'rm080030', 'bl160191', 'bo160176', 'at140305', 
	#'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	#'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ml160216', 'pb160320', 
	#'cc130066', 'in110286', 'ss120102'] #all subjects who have at least 60 total unseen target-present trials

filename = '_Train_loc_Test_AllSeen'

if ListAnalysis[0] is 'Loc_TrainLoc_TestLoc':
	beginTime = -0.096
	endTime = 0.8
else:
	beginTime = -0.2
	endTime = 3.496

tail = 1 #0 = 2-sided, 1 = 1-sided
chance = 0 #for analyses involving 
stat_params = 'permutation'
baselineCorr = True


#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding'

###Load decoding results (GAT, time)
for analysis in ListAnalysis:

	if baselineCorr:
		dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
		res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
	else:
		dat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/IndRes'
		res_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Stats'

	print('load: ' + analysis)
	all_scores = list() #initialize matrix containing all of GATs

	for subject in ListSubjects:
		score = np.load(dat_path + '/' + subject + filename + '-score.npy') #load actual data
		time = np.load(dat_path + '/' + subject + filename + '-time.npy') #load timing info
		all_scores.append(score) #matrix: NSubs x NTrain x NTest

		#Check whether time and data have the same axis
		if len(score[0]) != len(time):
			print('Updating and saving time axis now ...')

			tmp = np.where(time == beginTime)
			tmp = tmp[0][0]

			time = time[tmp :]

		#Check whether we also need to load the time axis for the localizer
		if (ListAnalysis[0] is 'Loc_Trainloc_TestAllSeen') or (ListAnalysis[0] is 'Resp_Trainloc_TestAllSeen') or (ListAnalysis[0] is 'Infer_Trainloc_TestAllSeen'):
			time_loc = np.load('/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/Loc_TrainLoc_TestLoc/GroupRes/Stats/Loc_TrainLoc_TestLoc-time.npy')
			np.save(res_path + '/' + analysis + '-time_loc.npy', time_loc)


	np.save(res_path + '/' + analysis + '-time.npy', time)
	np.save(res_path + '/' + analysis + '-all_scores.npy', all_scores)

	if stat_params is 'permutation':
		#Compute stats against theoretical chance level as obtained by permutations
		print('computing stats based on permutation: ' + analysis)

		p_values = myStats(np.array(all_scores) - chance, tail=tail) #p_values for entire gat (same shape as np.mean(gat))

		if ListAnalysis[0] is not 'Loc_Trainloc_TestAllSeen' and (ListAnalysis[0] is not 'Resp_Trainloc_TestAllSeen') and (ListAnalysis[0] is not 'Infer_Trainloc_TestAllSeen'):
			diag_offdiag = all_scores - np.tile([np.diag(sc) for sc in all_scores], [len(time), 1, 1]).transpose(1, 0, 2)
			p_values_off = myStats(diag_offdiag, tail=tail)

			scores_diag = [np.diag(sc) for sc in all_scores]
			p_values_diag = myStats(np.array(scores_diag)[:, :, None] - chance, tail=tail)
		elif ListAnalysis[0] is 'Loc_Trainloc_TestAllSeen' or (ListAnalysis[0] is 'Resp_Trainloc_TestAllSeen') or (ListAnalysis[0] is 'Infer_Trainloc_TestAllSeen'):
			slices = np.zeros([len(ListTois), len(ListSubjects), len(time)])
			p_values_slices = np.zeros([len(ListTois), len(time)])

			for ti, toi in enumerate(ListTois):
				all_scores = np.array(all_scores)
				tmp_begin = np.where(time_loc == toi[0])
				tmp_end = np.where(time_loc == toi[1])
				slices[ti, :, :] = np.mean(all_scores[:, tmp_begin[0][0] : tmp_end[0][0], :], axis=1)
				p_values_slices[ti, :] = myStats(np.array(slices[ti, :, :]) - chance, tail=tail)

	elif stat_params is 'Wilcoxon':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on uncorrected Wilcoxon: ' + analysis)

		p_values = parallel_stats(np.array(all_scores) - chance, correction=False) #p_values for entire gat (same shape as np.mean(gat))

		if (ListAnalysis[0] is not 'Loc_Trainloc_TestAllSeen') and (ListAnalysis[0] is not 'Resp_Trainloc_TestAllSeen') and (ListAnalysis[0] is not 'Infer_Trainloc_TestAllSeen') :
			diag_offdiag = all_scores - np.tile([np.diag(sc) for sc in all_scores], [len(time), 1, 1]).transpose(1, 0, 2)
			p_values_off = parallel_stats(np.array(diag_offdiag) - chance, correction=False) 

			scores_diag = [np.diag(sc) for sc in all_scores]
			p_values_diag = parallel_stats(np.array(scores_diag) - chance, correction=False) 

		elif (ListAnalysis[0] is 'Loc_Trainloc_TestAllSeen') or (ListAnalysis[0] is 'Resp_Trainloc_TestAllSeen') or (ListAnalysis[0] is 'Infer_Trainloc_TestAllSeen'):
			slices = np.zeros([len(ListTois), len(ListSubjects), len(time)])
			p_values_slices = np.zeros([len(ListTois), len(time)])

			for ti, toi in enumerate(ListTois):
				all_scores = np.array(all_scores)
				tmp_begin = np.where(time_loc == toi[0])
				tmp_end = np.where(time_loc == toi[1])
				slices[ti, :, :] = np.mean(all_scores[:, tmp_begin[0][0] : tmp_end[0][0], :], axis=1)
				p_values_slices[ti, :] = parallel_stats(np.array(slices[ti, :, :]) - chance, correction=False)

	elif stat_params is 'Wilcoxon-FDR':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on corrected Wilcoxon: ' + analysis)

		p_values = parallel_stats(np.array(all_scores) - chance, correction='FDR') #p_values for entire gat (same shape as np.mean(gat))

		if ListAnalysis[0] is not 'Loc_Trainloc_TestAllSeen':
			diag_offdiag = all_scores - np.tile([np.diag(sc) for sc in all_scores], [len(time), 1, 1]).transpose(1, 0, 2)
			p_values_off = parallel_stats(np.array(diag_offdiag) - chance, correction='FDR') 

			scores_diag = [np.diag(sc) for sc in all_scores]
			p_values_diag = parallel_stats(np.array(scores_diag) - chance, correction='FDR') 
		elif ListAnalysis[0] is 'Loc_Trainloc_TestAllSeen':
			slices = np.zeros([len(ListTois), len(ListSubjects), len(time)])
			p_values_slices = np.zeros([len(ListTois), len(time)])

			for ti, toi in enumerate(ListTois):
				all_scores = np.array(all_scores)
				tmp_begin = np.where(time_loc == toi[0])
				tmp_end = np.where(time_loc == toi[1])
				slices[ti, :, :] = np.mean(all_scores[:, tmp_begin[0][0] : tmp_end[0][0], :], axis=1)
				p_values_slices[ti, :] = parallel_stats(np.array(slices[ti, :, :]) - chance, correction='FDR')

	#Save
	if (ListAnalysis[0] is not 'Loc_Trainloc_TestAllSeen') and (ListAnalysis[0] is not 'Resp_Trainloc_TestAllSeen') and (ListAnalysis[0] is not 'Infer_Trainloc_TestAllSeen'):
		np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '-p_values.npy', p_values)
		np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '-p_values_off.npy', p_values_off)
		np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '-p_values_diag.npy', p_values_diag)
	else:
		np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '-p_values.npy', p_values)
		np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '-p_values_slices.npy', p_values_slices)
		np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '-slices.npy', slices)








