#Purpose: Plot diagonal decoding analyses for individual subjects (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 26 January 2018

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap

from menRot_plotGat import pretty_gat, pretty_decod, pretty_slices
from menRot_smooth import my_smooth

###Define important general variables###
ListAnalysis = ['Loc_TrainAllUnseen_TestAllUnseen']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

#ListSubjects = ['lm130479', 'am150105', 'nb140272', 'dp150209', 
	#'ag150338', 'ml140071', 'rm080030', 
	#'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 
	#'cs150204', 'mp110340', 'lg160230',   'ml160216', 'pb160320', 
	#'cc130066', 'in110286'] #subjects with at least 100 unseen trials

#ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	#'ag150338', 'ml140071', 'rm080030', 'bl160191', 'bo160176', 'at140305', 
	#'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	#'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ml160216', 'pb160320', 
	#'cc130066', 'in110286', 'ss120102'] #all subjects who have at least 60 total unseen target-present trials


BaselineCorr = True

chance = 0
smooth = True
smoothWindow = 2
stat_params = 'permutation'
tail = 0 #0 = 2-sided, 1 = 1-sided

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/TwoFolds'

###Plot diagonal decoding
for subi, sub in enumerate(ListSubjects):
	for condi, cond in enumerate(ListAnalysis):

		fig_diag, ax_diag = plt.subplots(1, 1, figsize=[4, 1.5])

		if BaselineCorr:
			dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
			stat_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
			res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Figures'
		else:
			dat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/IndRes'
			stat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Stats'
			res_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Figures'

		print('analysis: ' + cond)
		print('subject: ' + sub)

		#Load all necessary data
		time = np.load(stat_path + '/' + cond + '-time.npy') #load timing
		scores = np.array(np.load(stat_path + '/' + cond + '-all_scores.npy')) #load actual data 

		#Compute all other scores
		scores_diag = np.array([np.diag(sc) for sc in scores])

		#Plot
		plt.plot(time, scores_diag[subi, :])
		ax_diag.set_xlim(np.min(time), np.max(time))
		#pretty_decod(scores_diag[subi, :], times = time, sfreq = 125, sig = None, chance = chance, 
				#color = 'k', fill = None, ax = ax_diag)
		plt.show()

			