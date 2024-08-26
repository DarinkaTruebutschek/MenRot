#Purpose: Plot circular-linear correlation analyses (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 17 January 2018

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap

from menRot_plotGat import pretty_gat, pretty_decod, pretty_slices
from menRot_smooth import my_smooth

###Params to config###
channel = 'occipital_old'

chance = 0 #for analyses involving 
stat_alpha = 0.05 #statistical threshold
stat_params = 'permutation'
tail = 1 #0 = 2-sided, 1 = 1-sided

smooth = True
smoothWindow = 2


figs = list()

###Define important variables###
ListAnalysis = ['Loc2_Perm_report']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
ListFolder = ['loc2_circCorr_Perm_artifactRemoved']
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.7667], [1.8667, 2.0667], [2.0667, 3.2667], [3.2667, 3.496]] #time bins for which to display slices

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016'

###Define important parameters for figure###
contourPlot = True #regular depiction or just with contours
fig_alldiag = plt.figure(figsize=[8.27, 11.69], dpi=300)
axes_alldiag = gridspec.GridSpec(len(ListAnalysis), 1, hspace=0.1) #n_rows, n_columns

###Define colors (this will have to be facilitated)
def analysis(name, condition, query=None, title=None):
    return dict(name=name, condition=condition)

my_colors = (analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainLoc_TestLoc', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'))
cmap = plt.get_cmap('gist_rainbow')

for ii in range(len(ListTois)):
    color = np.array(cmap(float(ii)/len(ListTois)))
    my_colors[ii]['color']= color
    my_colors[ii]['cmap'] = LinearSegmentedColormap.from_list('RdBu', ['w', color, 'k'])

###Plot diagonal decoding and temporal generalization for each analysis
for anali, (my_analysis, ax_diag) in enumerate(zip(ListAnalysis, axes_alldiag)):

	#Get index for color (this should be fixed)
	if (my_analysis is 'Loc_TrainAll_TestAll') or (my_analysis is 'Loc_TrainRot_TestRot') or (my_analysis is 'Loc_TrainNoRot_TestNoRot'):
		tupIndex = 0
	elif my_analysis is 'Loc2_Perm_report':
		tupIndex = 1

	dat_path = path + '/' + ListFolder[anali] + '/Stats'
	stat_path = path + '/' + ListFolder[anali] + '/Stats'
	res_path = path + '/' + ListFolder[anali] + '/Figures/Timecourse'

	print('analysis: ' + my_analysis)

	#Load all necessary data
	time = np.load(stat_path + '/' + my_analysis + '_' + channel + '-time.npy') #load timing
	scores_diag = np.array(np.load(stat_path + '/' + my_analysis + '_' + channel + '-all_scores.npy')) #load actual data 

	p_values_diag = np.squeeze(np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '_' + channel + '-p_values_diag.npy')) #load p_values for diagonal

	#Compute one-sided p_value for diagonal if original stats were done with Wilcoxon
	if (stat_params is 'Wicoxon') and (tail == 1):
		p_values_diag = p_values_diag / 2.

	###Plot diagonal
	fig_diag, ax_diag = plt.subplots(1, 1, figsize=[4, 1.5])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores_diag]
		scores_diag = scores_smooth
		del scores_smooth

	pretty_decod(np.mean(scores_diag, axis=0), times = np.squeeze(time), sfreq = 250, sig = p_values_diag<stat_alpha, chance = chance, 
		color = my_colors[tupIndex]['color'], fill = True, ax = ax_diag)
	
	#Define ylim
	scores_diag = np.array(scores_diag)
	xlim, ylim = ax_diag.get_xlim(), np.array(ax_diag.get_ylim())
	sem = scores_diag.std(0)/np.sqrt(len(scores_diag))
	ylim = [np.min(scores_diag.mean(0)- sem), np.max(scores_diag.mean(0) + sem)]
	ax_diag.set_ylim(ylim)

	if (my_analysis is 'Loc_TrainAll_TestAll') or (my_analysis is 'Loc_TrainRot_TestRot') or (my_analysis is 'Loc_TrainNoRot_TestNoRot'):
		ax_diag.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
		ax_diag.axvline(3.2667, color='k', linewidth=1) #indicates response onset

		ax_diag.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
		ax_diag.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

		ax_diag.set_yticks([ylim[0], chance, ylim[1]])
		ax_diag.set_yticklabels(['', '', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})
	elif my_analysis is 'Loc2_Perm_report':
		if (ylim < 0) and (chance == 0):
			ax_diag.set_xticks(np.arange(0., 0.8, .2))
			ax_diag.set_xticklabels(['T', '0.2', '0.4', '0.6', '0.8'], fontdict={'family': 'arial', 'size': 12})

			ax_diag.set_yticks([ylim[0], chance, ylim[1]])
			ax_diag.set_yticklabels(['', '', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})
		elif (ylim > 0) and (chance == 0):
			ax_diag.set_xticks(np.arange(0., 0.8, .2))
			ax_diag.set_xticklabels(['T', '0.2', '0.4', '0.6', '0.8'], fontdict={'family': 'arial', 'size': 12})

			ax_diag.set_yticks([chance-0.005, chance, ylim[1]])
			ax_diag.set_yticklabels(['', '', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})

	plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-diag.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(fig_diag)
