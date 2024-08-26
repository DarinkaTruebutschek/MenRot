#Purpose: This script overlays the diagonal EOG and brain decoding on the same plot for individual subjects.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 29 October 2018

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap

from menRot_plotGat import pretty_gat, pretty_decod, pretty_slices
from menRot_smooth import my_smooth

###Define important general variables###
ListAnalysis = ['Loc_TrainAll_TestAll']

ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
ListExclude = ['av160302', 'bo160176', 'cb140229', 'mj100109', 'ml140071', 'mm140137', 'pb160320', 'ss120102'] #these subjects were selected based on visual inspection of both decoding time courses (first try)
ListExclude = ['ml140071', 'rm080030', 'bo160176', 'mm140137', 'av160302', 'pb160320', 'ss120102'] #these subjects were selected based on visual inspection of both decoding time courses (first try)
ListExclude = ['av160302', 'bo160176', 'cb140229',  'in110286', 'lg160230', 'mb160304', 'ml140071', 'mm140137', 'nb140272', 'pb160320', 'ss120102'] 
ListExclude = ['at140305', 'av160302', 'bo160176', 'cb140229', 'ef160362', 'in110286', 'lm130479', 'ml140071', 'mp110340', 'mp150285', 'nb140272', 'pb160320', 'rl130571', 'ss120102']

ListExclude = ['at140305', 'av160302', 'cb140229', 'cs150204', 'dp150209', 'in110286', 'mj100109',  'ml140071', 'mm140137', 'mp110340', 'pb160320', 'ss120102'] #resp 1
ListExclude = ['ag150338', 'am150105', 'at140305', 'av160302', 'cs150204', 'dp150209', 'in110286', 'mj100109', 'ml140071', 'mm140137', 'mp110340', 'pb160320', 'ss120102' ]

ListExclude=['am150105', 'at140305', 'bl160191', 'bo160176', 'cs150204', 'dp150209', 'lg160230', 'lk160274', 'mb160304', 'ml140071', 'ml160216', 'mm140137', 'nb140272', 'pb160320', 'rl130571', 'rm080030', 'ss120102'] #Infer with huge offset
ListExclude=['am150105', 'at140305', 'bl160191', 'bo160176', 'cs150204', 'dp150209', 'lg160230', 'lk160274', 'mb160304', 'ml140071', 'ml160216', 'mm140137', 'nb140272', 'pb160320', 'rl130571', 'rm080030', 'ss120102', 'av160302', 'cb140229'] #Infer with huge offset + little initial bump


#Construct second list with only included subjects
ind_excluded = [ListSubjects.index(sub) for sub in ListExclude]

ListFrequency = ['all']

BaselineCorr = True
chance = 0
vis = 'seen'

if ListFrequency[0] is 'all':
	smooth = True
	smoothWindow = 2
	brain_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore'
	eog_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/EOG'
else:
	smooth = False
	smoothWindow = 2
	path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore/TFA'

###Plot diagonal decoding for each subject and analysis###
for my_analysis in ListAnalysis:

	if BaselineCorr:
		stat_path_brain = brain_path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
		stat_path_eog = eog_path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
		res_path = eog_path + '/' + ListAnalysis[0] + '/GroupRes/Figures'

	#Load necessary data
	time = np.load(stat_path_brain + '/' + my_analysis + '-' + vis + '-time.npy')
	scores_brain = np.array(np.load(stat_path_brain + '/' + my_analysis + '-' + vis + '-all_scores.npy')) #load actual data 
	scores_eog = np.array(np.load(stat_path_eog + '/' + my_analysis + '-' + vis + '-all_scores.npy')) #load actual data 

	scores_diag_brain = np.array([np.diag(sc) for sc in scores_brain])
	scores_diag_eog = np.array([np.diag(sc) for sc in scores_eog])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores_diag_brain]
		scores_diag_brain = scores_smooth

		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores_diag_eog]
		scores_diag_eog = scores_smooth
		del scores_smooth

	for subi, subject in enumerate(ListSubjects):
		fig_diag, ax_diag = plt.subplots(1, 1, figsize=[6, 6])

		plt.hold(True)

		pretty_decod(scores_diag_brain[subi], times = time, sfreq = 125, sig = None, chance = chance, color = np.array([.64, .08, .18]), fill = False, ax = ax_diag)

		pretty_decod(scores_diag_eog[subi], times = time, sfreq = 125, sig = None, chance = chance, 
			color = np.array([.1, .1, .1]), fill = False, ax = ax_diag)

		ylim = [-1, 1]

		ax_diag.set_ylim(ylim)

		ax_diag.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
		ax_diag.axvline(3.2667, color='k', linewidth=1) #indicates response onset

		ax_diag.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
		ax_diag.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

		ax_diag.set_yticks([ylim[0], chance, ylim[1]])
		ax_diag.set_yticklabels(['%.2f' % ylim[0], '0.0', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})

		plt.savefig(res_path + '/' + my_analysis + subject + '_Comparison-EOGvsBrain_' + vis + '-diag.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
		
		plt.show()

###Compare what happens to brain and EOG decoding when excluding a certain set of subjects###
for my_analysis in ListAnalysis:
	scores_diag_brain_included = [scores_diag_brain[subi] for subi, sub in enumerate(ListSubjects) if subi not in ind_excluded]
	scores_diag_eog_included = [scores_diag_eog[subi] for subi, sub in enumerate(ListSubjects) if subi not in ind_excluded]

	fig_diag, ax_diag = plt.subplots(1, 1, figsize=[8, 3])

	plt.hold(True)

	#Plot excluded subjects
	pretty_decod(scores_diag_brain_included, times = time, sfreq = 125, sig = None, chance = chance, color = np.array([.64, .08, .18]), fill = False, ax = ax_diag)
	pretty_decod(scores_diag_eog_included, times = time, sfreq = 125, sig = None, chance = chance, color = np.array([.1, .1, .1]), fill = False, ax = ax_diag)

	#Plot all subjects
	pretty_decod(scores_diag_brain, times = time, sfreq = 125, sig = None, chance = chance, color = np.array([0.2, 0.3, 0.49]), fill = False, ax = ax_diag)
	pretty_decod(scores_diag_eog, times = time, sfreq = 125, sig = None, chance = chance, color = np.array([.5, .5, .5]), fill = False, ax = ax_diag)

	ylim = [-.1, .45]

	ax_diag.set_ylim(ylim)

	ax_diag.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
	ax_diag.axvline(3.2667, color='k', linewidth=1) #indicates response onset

	ax_diag.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	ax_diag.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

	ax_diag.set_yticks([ylim[0], chance, ylim[1]])
	ax_diag.set_yticklabels(['%.2f' % ylim[0], '0.0', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})

	plt.savefig(res_path + '/' + my_analysis + 'Group_Comparison-EOGvsBrain_' + vis + '-diag.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
		
	plt.show()