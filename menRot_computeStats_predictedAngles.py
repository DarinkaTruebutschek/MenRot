#Purpose: Compute between-subject statistics for analysis of predicted angles (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 18 December 2017

import sys

import numpy as np
import scipy.io as sio

from menRot_base import myStats, parallel_stats
from scipy import stats

###Define important variables###
ListAnalysis = ['Resp_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

n_bins = np.linspace(-np.pi, np.pi, 25)
vis = 'RightSeen'
tail = 1 #0 = 2-sided, 1 = 1-sided
chance = 1./24
#chance = 1./12 #for analyses involving 
stat_params = 'permutation'

beginTime = -0.2
endTime = 3.496

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
filename = path + vis + '_' + str(len(n_bins)) + '-histogram.npy'


###Load decoding results (GAT, time)
for analysis in ListAnalysis:

	all_hists = np.load(filename) #subjects * timebins * testing time

	print('load: ' + analysis)

	if stat_params is 'permutation':
		#Compute stats against theoretical chance level as obtained by permutations
		print('computing stats based on permutation: ' + analysis)

		p_values = myStats(np.array(all_hists) - chance, tail=tail) #p_values for entire gat (same shape as np.mean(gat))


	elif stat_params is 'Wilcoxon':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on uncorrected Wilcoxon: ' + analysis)

		p_values = parallel_stats(np.array(all_hists) - chance, correction=False) #p_values for entire gat (same shape as np.mean(gat))


	elif stat_params is 'Wilcoxon-FDR':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on corrected Wilcoxon: ' + analysis)

		p_values = parallel_stats(np.array(all_scores) - chance, correction='FDR') #p_values for entire gat (same shape as np.mean(gat))

	#Save
	np.save(path + '/' + analysis + '_' + stat_params +  str(tail) +  '_' + str(len(n_bins)) + '-' + vis + 'predictedAngles-p_values.npy', p_values)







