#Purpose: Plot decoding analyses (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 19 December 2017

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio

from menRot_plotGat import pretty_gat, pretty_decod, pretty_slices
from menRot_smooth import my_smooth

###Define important general variables###
ListAnalysis = ['Loc_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105']
#ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	#'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	#'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	#'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	#'cc130066', 'in110286', 'ss120102']
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.7667], [1.8667, 2.0667], [2.0667, 3.2667], [3.2667, 3.496]] #time bins for which to display slices

alpha = 0.05 #statistical threshold
chance = 0
smooth = True
smoothWindow = 2
stat_params = 'permutation'

figs = list()
table_toi = np.empty((len(ListAnalysis), len(ListTois)), dtype=object)
table_reversal = np.empty((len(ListAnalysis), 2), dtype=object)

stat_params = 'permutation'

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding'

###Define important parameters for figure###
contourPlot = True #regular depiction or just with contours
fig_alldiag = plt.figure(figsize=[6.5, 11], dpi=300)
axes_alldiag = gridspec.GridSpec(len(ListAnalysis), 1, hspace=0.1) #n_rows, n_columns

###Plot diagonal decoding and temporal generalization for each analysis
for ii, (analysis, ax_diag) in enumerate(zip(ListAnalysis, axes_alldiag)):

	dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
	stat_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
	res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Figures'

	print('analysis: ' + analysis)

	#Load all necessary data
	time = np.load(stat_path + '/' + analysis + '-time.npy') #load timing
	scores = np.load(stat_path + '/' + analysis + '-all_scores.npy') #load actual data 

	p_values = np.load(stat_path + '/' + analysis + '_' + stat_params + '-p_values.npy') #load p_values for gat
	p_values_off = np.load(stat_path + '/' + analysis + '_' + stat_params + '-p_values_off.npy') #load p_values for offdiag
	p_values_diag = np.squeeze(np.load(stat_path + '/' + analysis + '_' + stat_params + '-p_values_diag.npy')) #load p_values for diagonal

	#Compute all other scores
	diag_offdiag = scores - np.tile([np.diag(sc) for sc in scores], [len(time), 1, 1]).transpose(1, 0, 2)
	scores_diag = [np.diag(sc) for sc in scores]

	#Plot GAT
	clim = [chance, 0.1]
	fig_gat, ax_gat = plt.subplots(1, figsize=[7, 5.5])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores]
		scores = scores_smooth
		del scores_smooth

	plt.hold(True)
	pretty_gat(np.mean(scores, axis=0), times = time, chance = chance, ax = ax_gat, sig = None, cmap = 'coolwarm',
        clim = clim, colorbar = False, xlabel = 'Testing Time (s)', ylabel = 'Training Time (s)', sfreq = 125, diagonal = 'dimgray', test_times = None,
        contourPlot = contourPlot, steps = [0.025, 0.05, 0.075, 0.10, 0.125]) #indicates onset of cue
	
	#Axes props
	ax_gat.axvline(1.7667, color='k', linestyle ='dotted', linewidth=2)
	ax_gat.set_xticks(np.arange(0., 3.496, .5))
	ax_gat.set_xticklabels(['T', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], fontdict={'family': 'arial', 'size': 12})
	ax_gat.set_yticks(np.arange(0., 3.496, .5))
	ax_gat.set_yticklabels(['T', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], fontdict={'family': 'arial', 'size': 12})

	plt.savefig(res_path + '/' + analysis + '_' + stat_params + '-gat.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(fig_gat)
