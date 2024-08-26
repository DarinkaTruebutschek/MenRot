#Purpose: Compute between-subject statistics for decoding analyses with Riemannian decoder (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 02 May 2018

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap

from menRot_base import myStats, parallel_stats
from plotBox import plotBox
from plotBar import plotBar
from scipy import stats

###Define important variables###
ListAnalysis = ['Resp_TrainAll_TestAll']
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.768], [1.768, 3.268], [3.268, 3,5],
	[1.77, 1.87], [1.87, 1.97], [1.97, 2.07], [2.07, 2.17], [2.17, 2.27], [2.27, 2.37], [2.37, 2.47], [2.47, 2.57], [2.57, 2.67], [2.67, 2.77], [2.77, 2.87], [2.87, 2.97],
	[2.97, 3.07], [3.07, 3.17], [3.17, 3.27]]
if ListAnalysis[0] is not 'Infer_TrainnoRot_TestnoRot' and ListAnalysis[0] is not 'Resp_TrainnoRot_TestnoRot':
	ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
		'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
		'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
		'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
		'cc130066', 'in110286', 'ss120102']
elif ListAnalysis[0] is 'Infer_TrainnoRot_TestnoRot':
	ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
		'ag150338', 'ml140071', 'bl160191', 'lj150477','bo160176', 'at140305', 
		'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
		'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
		'cc130066']
elif ListAnalysis[0] is 'Resp_TrainnoRot_TestnoRot':
		ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
		'ag150338', 'ml140071', 'bl160191', 'lj150477','bo160176', 'at140305', 
		'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
		'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
		'cc130066']
#ListSubjects = ['lm130479', 'am150105']

vis = 'NoRotUnseenIncorr'
tail = 0 #0 = 2-sided, 1 = 1-sided
chance = 0 #for analyses involving 
stat_params = 'Wilcoxon'

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Riemann/'
filename = '_Train_All_Test_All'

###Load decoding results (GAT, time)
for analysis in ListAnalysis:
	dat_path = path + ListAnalysis[0] + '/IndRes'
	res_path = path + ListAnalysis[0] + '/GroupRes/Stats'
	fig_path = path + ListAnalysis[0] + '/GroupRes/Figures'

	print('load: ' + analysis)
	all_scores = list() #initialize matrix containing decoding scores for all time_bins

	for subject in ListSubjects:
		if vis is 'seen':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreSeen.npy') #load actual data
			tupIndex = np.array([.64, .08, .18])
			ymax = 1.2 #for loc
			ymin = -.6
			#ymin = None
			#ymax = None
		elif vis is 'unseen':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreUnseen.npy') #load actual data
			tupIndex = np.array([.2, .3, .49])
			ymax = 1.2
			ymin = -.6
			#ymin = None
			#ymax = None
		elif vis is 'unseenCorr':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreUnseenCorr.npy') #load actual data
			tupIndex = np.array([0, .45, .74])
			ymax = 1.2
			ymin = -.6
			#ymin = None
			#ymax = None
		elif vis is 'unseenIncorr':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreUnseenIncorr.npy') #load actual data
			tupIndex = np.array([.39, .47, .64])
			ymax = 1.2
			ymin = -.6
			#ymin = None
			#ymax = None
		elif vis is 'RotSeen':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreRotSeen.npy') #load actual data
			tupIndex = np.array([.64, .08, .18])
			ymin = -.75
			ymax = 1.0
		elif vis is 'RotUnseen':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreRotUnseen.npy') #load actual data
			tupIndex = np.array([.2, .3, .49])
			ymin = -.75
			ymax = 1.0
		elif vis is 'RotUnseenCorr':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreRotUnseenCorr.npy') #load actual data
			tupIndex = np.array([0, .45, .74])
			ymin = -.75
			ymax = 1.0
		elif vis is 'RotUnseenIncorr':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreRotUnseenIncorr.npy') #load actual data
			tupIndex = np.array([.39, .47, .64])
			ymin = -.75
			ymax = 1.0
		elif vis is 'NoRotSeen':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreNoRotSeen.npy') #load actual data
			tupIndex = np.array([.64, .08, .18])
			ymin = -0.85
			ymax = 1.3
		elif vis is 'NoRotUnseen':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreNoRotUnseen.npy') #load actual data
			tupIndex = np.array([.39, .47, .64])
			ymin = -0.85
			ymax = 1.3
		elif vis is 'NoRotUnseenCorr':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreNoRotUnseenCorr.npy') #load actual data
			tupIndex = np.array([0, .45, .74])
			ymin = -0.85
			ymax = 1.3
		elif vis is 'NoRotUnseenIncorr':
			score = np.load(dat_path + '/' + subject + filename + '_all-scoreNoRotUnseenIncorr.npy') #load actual data
			tupIndex = np.array([.39, .47, .64])
			ymin = -0.85
			ymax = 1.3

		#Check whether data exists for this subject
		if np.unique(np.isnan(np.unique(score))): 
			print('Skipping subject: ' + subject)
		else:
			all_scores.append(score) #matrix: NSubs x NTrain x NTest

	np.save(res_path + '/' + analysis + '_' + vis + '-time.npy', ListTois)
	np.save(res_path + '/' + analysis + '_' + vis + '-all_scores.npy', all_scores)

	#Compute stats using uncorrected Wilcoxon signed-rank test
	if stat_params is 'Wilcoxon':
		print('computing stats based on uncorrected Wilcoxon: ' + analysis)
		p_values = parallel_stats(np.array(all_scores) - chance, correction='FDR') #p_values for entire gat (same shape as np.mean(gat))

	np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + vis + '-p_values.npy', p_values)

	#Plot bar
	#Extract relevant data
	dat2plot1 = np.zeros((len(ListSubjects), 6))
	dat2plot2 = np.zeros((len(ListSubjects), 15))
	for subi, subject in enumerate(ListSubjects):
		dat2plot1[subi, :] = all_scores[subi][0:6]
		dat2plot2[subi, :] = all_scores[subi][6:21]

	ax = plotBar(np.mean(dat2plot1, axis=0), stats.sem(dat2plot1), p_values[0 : 6], tupIndex)
	plt.savefig(fig_path + '/' + analysis + vis + '_entireEpoch-barPlot.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(ax)

	ax = plotBar(np.mean(dat2plot2, axis=0), stats.sem(dat2plot2), p_values[6 : 21], tupIndex)
	plt.savefig(fig_path + '/' + analysis + vis + '_delay2-barPlot.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(ax)

	#Save data
	my_table = [np.mean(all_scores, axis = 0).T, stats.sem(all_scores).T, p_values.T]

	with open(res_path + '/' + analysis + '_' + vis + '-Table.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = '\t')
		[writer.writerow(r) for r in my_table]

	#Box plot	
	ax = plotBox(dat2plot1, stats.sem(dat2plot1), p_values[0 : 6], tupIndex, ymax, ymin)
	plt.savefig(fig_path + '/' + analysis + vis + '_entireEpoch-BoxPlot.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(ax)

	ax = plotBox(dat2plot2, stats.sem(dat2plot2), p_values[6 : 21], tupIndex, ymax, ymin)
	plt.savefig(fig_path + '/' + analysis + vis + '_delay2-BoxPlot.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(ax)