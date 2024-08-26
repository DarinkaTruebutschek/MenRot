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
ListAnalysis = ['Train_All_Test_All']                                                                                                                                                                                  
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
ListFrequency = ['HighBeta']

ListTois = [[-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.7667], [1.8667, 2.0667], [2.0667, 3.2667], [3.2667, 3.496], [-0.2, 0], [0.1, 0.3], [0.3, 0.6], [0.6, 1.7667], [1.8667, 2.0667], [2.0667, 3.2667], [3.2667, 3.496]] #time bins for which to display slices
ListSlices = [[0.096, 0.296], [0.296, 0.6], [0.6, 0.8], [0.296, 0.8], [0.096, 0.8]]

BaselineCorr = True

stat_alpha = 0.05 #statistical threshold
chance = 0.5

if ListFrequency[0] is 'all':
	smooth = True
	smoothWindow = 2
	path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis'
else:
	smooth = False
	smoothWindow = 2
	path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis_TFA'


stat_params = 'permutation'
tail = 1#0 = 2-sided, 1 = 1-sided

figs = list()
table_toi = np.empty((len(ListAnalysis), len(ListTois)), dtype=object)
table_reversal = np.empty((len(ListAnalysis), 2), dtype=object)

###Define important parameters for figure###
contourPlot = True #regular depiction or just with contours
fig_alldiag = plt.figure(figsize=[8.27, 11.69], dpi=300)
axes_alldiag = gridspec.GridSpec(len(ListAnalysis), 1, hspace=0.1) #n_rows, n_columns

###Define colors (this will have to be facilitated)
def analysis(name, condition, query=None, title=None):
    return dict(name=name, condition=condition)

my_colors = (analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainLoc_TestLoc', condition='loc'),
	analysis(name='Loc_TrainAllSeen_TestAllSeen', condition='loc'), analysis(name='Loc_TrainAllUnseen_TestAllUnseen', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainLoc_TestLoc', condition='loc'),
	analysis(name='Loc_TrainAllSeen_TestAllSeen', condition='loc'), analysis(name='Loc_TrainAllUnseen_TestAllUnseen', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'), analysis(name='Loc_TrainAll_TestAll', condition='loc'),
	analysis(name='Loc_TrainAll_TestAll', condition='loc'))
cmap = plt.get_cmap('gist_rainbow')

for ii in range(len(ListTois)):
    color = np.array(cmap(float(ii)/len(ListTois)))
    my_colors[ii]['color']= color
    my_colors[ii]['cmap'] = LinearSegmentedColormap.from_list('RdBu', ['w', color, 'k'])
    
###Plot diagonal decoding and temporal generalization for each analysis
for ii, (my_analysis, ax_diag) in enumerate(zip(ListAnalysis, axes_alldiag)):

	#Get index for color (this should be fixed)
	#tupIndex = 12 #1 = orange, 2 = yellow, 3 = bright yellow, 4= green/yellow, 5 = 6 = bright green, 7 = light green, 8 = light blue, 9 = darkblue, 10 = even darker blue, 11 = almost violet, 12  = dark violet, 13 = pink
	tupIndex = 9
	#clim = [chance, 0.1]
	#steps = [0.025, 0.05, 0.075, 0.1, 0.125]
	#clim = [chance, 0.08]
	#steps = [0.04, 0.05, 0.06, 0.08, 0.09]
	#clim = [chance, 0.63]
	#steps = [0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63] #all

	#clim = [chance, 0.6]
	#steps = np.linspace(0.53, 0.6, 7) #noRot

	#clim = [chance, 0.6]
	#steps = np.linspace(0.51, 0.6, 7) #Rot

	#clim = [chance, 0.63]
	#steps = np.linspace(0.52, 0.63, 7) #Cross-gen

	#clim = [chance, 0.55]
	#steps = np.linspace(0.51, 0.54, 4) #UnseenCorr - UnseenIncorr

	#clim = [chance, 0.56]
	#steps = np.linspace(0.51, 0.56, 6) #UnseenCorr - UnseenIncorr (rot)

	#clim = [chance, 0.59]
	#steps = np.linspace(0.51, 0.59, 6) #UnseenCorr - UnseenIncorr (no Rot)

	#clim = [chance, 0.63]
	#steps = np.linspace(0.52, 0.63, 7) #Seen - UnseenIncorr

	clim = [chance, 0.56]
	steps = np.linspace(0.52, 0.56, 7) #alpha band

	if BaselineCorr:
		dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
		stat_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
		res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Figures'
	else:
		dat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/IndRes'
		stat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Stats'
		res_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Figures'

	print('analysis: ' + my_analysis)

	#Load all necessary data
	if ListFrequency[0] is 'all':
		time = np.load(stat_path + '/' + my_analysis + '-time.npy') #load timing
		scores = np.array(np.load(stat_path + '/' + my_analysis + '-all_scores.npy')) #load actual data 

		p_values = np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) +  '-p_values.npy') #load p_values for gat
		p_values_off = np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-p_values_off.npy') #load p_values for offdiag
		p_values_diag = np.squeeze(np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-p_values_diag.npy')) #load p_values for diagonal
	else:
		time = np.load(stat_path + '/' + my_analysis + '_' + ListFrequency[0] + '-time.npy') #load timing
		scores = np.array(np.load(stat_path + '/' + my_analysis + '_' + ListFrequency[0] + '-all_scores.npy')) #load actual data 

		p_values = np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '_' + ListFrequency[0] + '-p_values.npy') #load p_values for gat
		p_values_off = np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '_' + ListFrequency[0] + '-p_values_off.npy') #load p_values for offdiag
		p_values_diag = np.squeeze(np.load(stat_path + '/' + my_analysis + '_' + stat_params + str(tail) + '_' + ListFrequency[0] + '-p_values_diag.npy')) #load p_values for diagonal

	#Compute all other scores
	diag_offdiag = np.array(scores - np.tile([np.diag(sc) for sc in scores], [len(time), 1, 1]).transpose(1, 0, 2))
	scores_diag = np.array([np.diag(sc) for sc in scores])

	#Compute one-sided p_value for diagonal if original stats were done with Wilcoxon
	if (stat_params is 'Wilcoxon') and (tail == 1):
		p_values_off = p_values_off / 2.
		p_values_diag = p_values_diag / 2.

	###Plot GAT
	fig_gat, ax_gat = plt.subplots(1, 1, figsize=[5, 4])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores]
		scores = scores_smooth
		del scores_smooth 

	#Determine whether or not there are any signifcant values
	if len(np.unique(p_values < stat_alpha)) == 1: #either all or none significant
		sig = None
	else:
		sig = p_values < stat_alpha

	pretty_gat(np.mean(scores, axis=0), times = time, chance = chance, ax = ax_gat, sig = sig , cmap = 'coolwarm', clim = clim, 
		colorbar = False, xlabel = 'Testing Time (s)', ylabel = 'Training Time (s)', sfreq = 125, diagonal = 'dimgray', 
		test_times = None, contourPlot = contourPlot, steps = steps) #indicates onset of cue

	ax_gat.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
	ax_gat.axhline(1.7667, color='k', linewidth=1)

	ax_gat.axvline(3.2667, color='k', linewidth=1) #indicates response onset
	ax_gat.axhline(3.2667, color='k', linewidth=1)

	ax_gat.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	ax_gat.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

	ax_gat.set_yticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	ax_gat.set_yticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

	ax_gat.set_aspect('equal')

	if ListFrequency[0] is 'all':
		plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-gat.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	else:
		plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '_' + ListFrequency[0] + '-gat.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')

	plt.show(fig_gat)

	###Plot diagonal
	fig_diag, ax_diag = plt.subplots(1, 1, figsize=[4, 1.5])

	if smooth:
		scores_smooth = [my_smooth(sc, smoothWindow) for sc in scores_diag]
		scores_diag = scores_smooth
		del scores_smooth

	pretty_decod(scores_diag, times = time, sfreq = 125, sig = p_values_diag<stat_alpha, chance = chance, 
		color = np.array([0.64, 0.08, 0.18]), fill = True, ax = ax_diag) #for vis
	#pretty_decod(scores_diag, times = time, sfreq = 125, sig = p_values_diag<stat_alpha, chance = chance, 
		#color = my_colors[tupIndex]['color'], fill = True, ax = ax_diag) 
	
	#Define ylim
	scores_diag = np.array(scores_diag)
	xlim, ylim = ax_diag.get_xlim(), np.array(ax_diag.get_ylim())
	sem = scores_diag.std(0)/np.sqrt(len(scores_diag))
	ylim = [np.min(scores_diag.mean(0)- sem), np.max(scores_diag.mean(0) + sem)]
	#ylim = [.48, .64] #all
	#ylim = [.46, .62] #noRot
	#ylim = [.48, .60] #Rot
	#ylim = [-0.05, 0.17] #for target
	#ylim = [.46, .65] #for Cross-gen
	#ylim = [.45, .64] #to compare seen vs unseen, correct, incorrect
	#ylim = [.46, .58] #alpha band
	#ylim = [.44, .58] #beta band
	ylim = [.46, .58] # high beta band

	ax_diag.set_ylim(ylim)

	ax_diag.axvline(1.7667, color='k', linewidth=1) #indicates cue onset
	ax_diag.axvline(3.2667, color='k', linewidth=1) #indicates response onset

	ax_diag.set_xticks(np.sort(np.append(np.arange(0., 3.496, .5), np.array([1.776, 3.2667]))))
	ax_diag.set_xticklabels(['T', '0.5', '1.0', '1.5', 'C', '2.0', '2.5', '3.0', 'R'], fontdict={'family': 'arial', 'size': 12})

	ax_diag.set_yticks([ylim[0], chance, ylim[1]])
	ax_diag.set_yticklabels(['', '', '%.2f' % ylim[1]], fontdict={'family': 'arial', 'size': 12})
		
	if ListFrequency[0] is 'all':
		plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '-diag.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	else:
		plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) + '_' + ListFrequency[0] + '-diag.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(fig_diag)
	