#Purpose: Compute and plot decoding performance averaged over timeBins.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 03 May 2018

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
ListAnalysis = ['Loc_TrainAll_TestAll']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']

#ListTois = [[-0.2, 0], [0.104, 0.304], [0.304, 0.6], [0.6, 1.76], [1.76, 3.264], [3.264, 3.496]] #time bins for which to display slices
ListTois = ([[1.76, 1.864], [1.864, 1.96], [1.96, 2.064], [2.064, 2.16], [2.16, 2.264], [2.264, 2.36], [2.36, 2.464], 
	[2.464, 2.56], [2.56, 2.664], [2.664, 2.76], [2.76, 2.864], [2.864, 2.96], [2.96, 3.064], [3.064, 3.16], [3.16, 3.264], [3.264, 3.36], [3.36, 3.464]])
ListTois = ([1.0, 1.104], [1.104, 1.20], [1.2, 1.304], [1.304, 1.4], [1.4, 1.504], [1.504, 1.60], [1.6, 1.704])
vis = 'unseen'

BaselineCorr = True
stat_params = 'permutation'
tail = 1#0 = 2-sided, 1 = 1-sided
chance = 0

dat2plot = np.zeros((len(ListAnalysis), len(ListTois), len(ListSubjects)))
sem2plot = np.zeros((len(ListAnalysis), len(ListTois)))
p_values = np.zeros((len(ListAnalysis), len(ListTois)))
stat = np.zeros((len(ListAnalysis), len(ListTois)))

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/Subscore'

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
	time = np.load(stat_path + '/' + my_analysis + '-' + vis + '-time.npy') #load timing
	scores = np.array(np.load(stat_path + '/' + my_analysis + '-' + vis + '-all_scores.npy')) #load actual data 
	scores_diag = np.array([np.diag(sc) for sc in scores])

	for toi_i, toi in enumerate(ListTois):
		print(toi)
		(dat2plot[analysis_i][toi_i], sem2plot[analysis_i][toi_i], p_values[analysis_i][toi_i], stat[analysis_i][toi_i]) = extractAverages(scores_diag, time, toi, chance, 'Wilcoxon')

	#Plot bar
	ax = plotBar(np.mean(dat2plot, axis = 2), sem2plot, p_values, np.array([0.64, 0.08, 0.18]))
	plt.savefig(res_path + '/' + my_analysis + '-' + vis + '-barPlot.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
	plt.show(ax)

	#Save data
	np.save(stat_path + '/' + my_analysis + '-meanDat.npy', np.mean(dat2plot, axis = 2))
	np.save(stat_path + '/' + my_analysis + '-semDat.npy', sem2plot)
	np.save(stat_path + '/' + my_analysis + '-pvalDat.npy', p_values)

	my_table = [np.mean(dat2plot, axis = 2).T, sem2plot.T, p_values.T]
	#my_table2 = [timei for timei in dat2plot.T[0]]
	my_table2 = [dat2plot[0][t].T for t, timei in enumerate(dat2plot.T[0])]
	my_table2 = zip(*my_table2)

	with open(stat_path + '/' + my_analysis + '-' + vis + '-Table.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = '\t')
		[writer.writerow(r) for r in my_table]

	with open(stat_path + '/' + my_analysis + '-' + vis + '-Table2.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = '\t')
		[writer.writerow(r) for r in my_table2]
