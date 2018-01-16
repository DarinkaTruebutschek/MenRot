#Purpose: Plot decoding analyses (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 19 December 2017

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
#ListSubjects = ['lm130479', 'am150105', 'cb140229']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.7667], [1.8667, 2.0667], [2.0667, 3.2667], [3.2667, 3.496]] #time bins for which to display slices

stat_alpha = 0.05 #statistical threshold
chance = 0
smooth = True
smoothWindow = 2
stat_params = 'permutation'
tail = 0 #0 = 2-sided, 1 = 1-sided

figs = list()
table_toi = np.empty((len(ListAnalysis), len(ListTois)), dtype=object)
table_reversal = np.empty((len(ListAnalysis), 2), dtype=object)

stat_params = 'permutation'

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding'

###Define important parameters for figure###
contourPlot = True #regular depiction or just with contours
fig_alldiag = plt.figure(figsize=[8.27, 11.69], dpi=300)
axes_alldiag = gridspec.GridSpec(len(ListAnalysis), 1, hspace=0.1) #n_rows, n_columns

###Define colors (this will have to be facilitated)
def analysis(name, condition, query=None, title=None):
    return dict(name=name, condition=condition)

my_colors = (analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'))
cmap = plt.get_cmap('gist_rainbow')

for ii in range(len(ListTois)):
    color = np.array(cmap(float(ii)/len(ListTois)))
    my_colors[ii]['color']= color
    my_colors[ii]['cmap'] = LinearSegmentedColormap.from_list('RdBu', ['w', color, 'k'])
    
###Plot diagonal decoding and temporal generalization for each analysis
for ii, (my_analysis, ax_diag) in enumerate(zip(ListAnalysis, axes_alldiag)):

	dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
	stat_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
	res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Figures'

	print('analysis: ' + my_analysis)

	#Load all necessary data
	time = np.load(stat_path + '/' + my_analysis + '-time.npy') #load timing
	scores = np.array(np.load(stat_path + '/' + my_analysis + '-all_scores.npy')) #load actual data 

	p_values = np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) +  '-p_values.npy') #load p_values for gat
	p_values_off = np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-p_values_off.npy') #load p_values for offdiag
	p_values_diag = np.squeeze(np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-p_values_diag.npy')) #load p_values for diagonal

	#Compute all other scores
	diag_offdiag = np.array(scores - np.tile([np.diag(sc) for sc in scores], [len(time), 1, 1]).transpose(1, 0, 2))
	scores_diag = np.array([np.diag(sc) for sc in scores])

	###Plot GAT
	clim = [chance, 0.1]
	fig_gat, ax_gat = plt.subplots(1, 1, figsize=[5, 4])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores]
		scores = scores_smooth
		del scores_smooth

	#plt.hold(True)
	pretty_gat(np.mean(scores, axis=0), times = time, chance = chance, ax = ax_gat, sig = None, cmap = 'coolwarm',
        clim = clim, colorbar = False, xlabel = 'Testing Time (s)', ylabel = 'Training Time (s)', sfreq = 125, diagonal = 'dimgray', test_times = None,
        contourPlot = contourPlot, steps = [0.025, 0.05, 0.075, 0.10, 0.125]) #indicates onset of cue
	
	#Axes props
	ax_gat.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
	ax_gat.axhline(1.7667, color='k', linewidth=1)

	ax_gat.axvline(3.2667, color='k', linewidth=1) #indicates response onset
	ax_gat.axhline(3.2667, color='k', linewidth=1)

	ax_gat.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	ax_gat.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})
	#ax_gat.set_xticks(np.arange(0., 3.496, .5))

	ax_gat.set_yticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	#ax_gat.set_yticks(np.arange(0., 3.496, .5))
	ax_gat.set_yticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

	ax_gat.set_aspect('equal')

	plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-gat.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(fig_gat)
	#plt.hold(False)

	###Plot diagonal
	fig_diag, ax_diag = plt.subplots(1, 1, figsize=[4, 1.5])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores_diag]
		scores_diag = scores_smooth
		del scores_smooth

	pretty_decod(np.mean(scores_diag, axis=0), times = time, sfreq = 125, sig = p_values_diag<stat_alpha, chance = chance, 
		color = my_colors[0]['color'], fill = True, ax = ax_diag)
	
	#Define ylim
	scores_diag = np.array(scores_diag)
	xlim, ylim = ax_diag.get_xlim(), np.array(ax_diag.get_ylim())
	sem = scores_diag.std(0)/np.sqrt(len(scores_diag))
	ylim = [np.min(scores_diag.mean(0)- sem), np.max(scores_diag.mean(0) + sem)]
	ax_diag.set_ylim(ylim)

	ax_diag.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
	ax_diag.axvline(3.2667, color='k', linewidth=1) #indicates response onset

	ax_diag.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	ax_diag.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

	ax_diag.set_yticks([ylim[0], chance, ylim[1]])
	ax_diag.set_yticklabels(['', '', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})

	plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-diag.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(fig_diag)

	###Plot slices
	fig_offdiag, ax_offdiag = plt.subplots(len(ListTois), 1, figsize=[5, 6])

	pretty_slices(scores, times = time, chance = chance, axes = ax_offdiag, sfreq = 125,
		sig = p_values < stat_alpha, sig_diagoff = p_values_off < stat_alpha, colors = ['k', 'b', 'k', 'b', 'k', 'b', 'k', 'b'], tois = np.arange(-0.2, 3.496, 0.2), 
		fill_color = 'yellow')