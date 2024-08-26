#Purpose: Compute and plot decoding performance averaged over timeBins.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 12 March 2018

###Setup
import sys

import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap
from extractAverages import extractAverages
from plotBar import plotBar

###Define important variables
ListAnalysis = ['Train_AllUnseenAcc_Test_AllUnseenAcc']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
#ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	#'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	#'mm140137', 'lk160274', 'av160302', 'cc150418', 
	#'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	#'cc130066', 'in110286', 'ss120102'] #only subjects with sufficient blindsight across all conditions
ListTois = [[-0.2, 0], [0.104, 0.304], [0.304, 0.6], [0.6, 1.76], [1.76, 3.264], [3.264, 3.496]] #time bins for which to display slices

BaselineCorr = True
chance = 0.5

dat2plot = np.zeros((len(ListAnalysis), len(ListTois), len(ListSubjects)))
sem2plot = np.zeros((len(ListAnalysis), len(ListTois)))
p_values = np.zeros((len(ListAnalysis), len(ListTois)))
stat = np.zeros((len(ListAnalysis), len(ListTois)))

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis'

if BaselineCorr:
	dat_path = path + '/' + ListAnalysis[0] + '/IndRes'
	stat_path = path + '/' + ListAnalysis[0] + '/GroupRes/Stats'
	res_path = path + '/' + ListAnalysis[0] + '/GroupRes/Figures'
else:
	dat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/IndRes'
	stat_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Stats'
	res_path = path + '/NoBaseline/' + ListAnalysis[0] + '/GroupRes/Figures'

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

for ii in np.arange(14):
    color = np.array(cmap(float(ii)/14))
    my_colors[ii]['color']= color
    my_colors[ii]['cmap'] = LinearSegmentedColormap.from_list('RdBu', ['w', color, 'k'])

###Load all necessary data
for analysis_i, my_analysis in enumerate(ListAnalysis):
	time = np.load(stat_path + '/' + my_analysis + '-time.npy') #load timing
	scores = np.array(np.load(stat_path + '/' + my_analysis + '-all_scores.npy')) #load actual data 
	scores_diag = np.array([np.diag(sc) for sc in scores])

	for toi_i, toi in enumerate(ListTois):
		print(toi)
		(dat2plot[analysis_i][toi_i], sem2plot[analysis_i][toi_i], p_values[analysis_i][toi_i], stat[analysis_i][toi_i]) = extractAverages(scores_diag, time, toi, chance, 'Wilcoxon')

	#Plot bar
	ax = plotBar(np.mean(dat2plot, axis = 2), sem2plot, p_values, np.array([0.64, 0.08, 0.18]))
	plt.savefig(res_path + '/' + my_analysis + '-barPlot.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(ax)

	#Save data
	np.save(stat_path + '/' + my_analysis + '-meanDat.npy', np.mean(dat2plot, axis = 2))
	np.save(stat_path + '/' + my_analysis + '-semDat.npy', sem2plot)
	np.save(stat_path + '/' + my_analysis + '-pvalDat.npy', p_values)

	my_table = [np.mean(dat2plot, axis = 2).T, sem2plot.T, p_values.T]

	with open(stat_path + '/' + my_analysis + '-Table.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = '\t')
		[writer.writerow(r) for r in my_table]
