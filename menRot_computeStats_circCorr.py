#Purpose: Compute between-subject statistics for circular-linear correlation analyses 
# run in Matlab (adapted from J.R. King).
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 17 January 2018

import sys

import numpy as np
import scipy.io as sio

from menRot_base import myStats, parallel_stats
from scipy import stats

###Params to config###
channel = 'occipital_old'
tail = 1 #0 = 2-sided, 1 = 1-sided
chance = 0 #for analyses involving 
stat_params = 'permutation'

###Define important variables###
ListAnalysis = ['Loc2_Perm_report']
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
ListFolder = ['loc2_circCorr_Perm_artifactRemoved']

if channel is 'occipital_old':
	ListSensors = ['MEG1713', 'MEG1712', 'MEG1722', 'MEG1723', 'MEG1732', 'MEG1733', 'MEG1743', 
    	'MEG1742', 'MEG2113', 'MEG2112', 'MEG2122', 'MEG2123', 'MEG2133', 'MEG2132', 'MEG2143', 'MEG2142',
    	'MEG2512', 'MEG2513', 'MEG2522', 'MEG2523', 'MEG2533', 'MEG2532', 'MEG2543', 'MEG2542']
elif channel is 'temp_left':
	ListSensors = ['MEG1512', 'MEG1513', 'MEG1522', 'MEG1523', 'MEG1533', 'MEG1532', 'MEG1543',	
		'MEG1542', 'MEG1613', 'MEG1612', 'MEG1622',	'MEG1623', 'MEG1632', 'MEG1633', 'MEG1643',
		'MEG1642', 'MEG1912', 'MEG1913', 'MEG1923',	'MEG1922', 'MEG1932', 'MEG1933', 'MEG1943',	'MEG1942']
elif channel is 'temp_right':
	ListSensors = ['MEG2312', 'MEG2313', 'MEG2323',	'MEG2322', 'MEG2332', 'MEG2333', 'MEG2343',	
		'MEG2342', 'MEG2412', 'MEG2413', 'MEG2423',	'MEG2422', 'MEG2433', 'MEG2432', 'MEG2442',
		'MEG2443', 'MEG2612', 'MEG2613', 'MEG2623',	'MEG2622', 'MEG2633', 'MEG2632', 'MEG2642',	'MEG2643']
elif channel is 'frontal':
	ListSensors = ['MEG0113', 'MEG0112', 'MEG0122',	'MEG0123', 'MEG0132', 'MEG0133', 'MEG0143',	'MEG0142',
		'MEG0313', 'MEG0312', 'MEG0322', 'MEG0323',	'MEG0333', 'MEG0332', 'MEG0343', 'MEG0342',	'MEG0513',
		'MEG0512', 'MEG0523', 'MEG0522', 'MEG0532',	'MEG0533', 'MEG0542', 'MEG0543', 'MEG0813',	'MEG0812',
		'MEG0822', 'MEG0823', 'MEG0913', 'MEG0912',	'MEG0923', 'MEG0922', 'MEG0932', 'MEG0933',	'MEG0942',
		'MEG0943', 'MEG1213', 'MEG1212', 'MEG1223', 'MEG1222', 'MEG1232', 'MEG1233', 'MEG1243',	'MEG1242',
		'MEG1412', 'MEG1413', 'MEG1423', 'MEG1422',	'MEG1433', 'MEG1432', 'MEG1442', 'MEG1443']
elif channel is 'all_post':
	ListSensors = ['MEG0713', 'MEG0712', 'MEG0723',	'MEG0722', 'MEG0733', 'MEG0732', 'MEG0743',	'MEG0742',
		'MEG1512', 'MEG1513', 'MEG1522', 'MEG1523',	'MEG1533', 'MEG1532', 'MEG1543', 'MEG1542',	'MEG1613',	
		'MEG1612', 'MEG1622', 'MEG1623', 'MEG1632',	'MEG1633', 'MEG1643', 'MEG1642', 'MEG1713',	'MEG1712',
		'MEG1722', 'MEG1723', 'MEG1732', 'MEG1733',	'MEG1743', 'MEG1742', 'MEG1813', 'MEG1812',	'MEG1822',
		'MEG1823', 'MEG1832', 'MEG1833', 'MEG1843',	'MEG1842', 'MEG1912', 'MEG1913', 'MEG1923',	'MEG1922',
		'MEG1932', 'MEG1933', 'MEG1943', 'MEG1942',	'MEG2013', 'MEG2012', 'MEG2023', 'MEG2022',	'MEG2032',
		'MEG2033', 'MEG2042', 'MEG2043', 'MEG2113',	'MEG2112', 'MEG2122', 'MEG2123', 'MEG2133',	'MEG2132',
		'MEG2143', 'MEG2142', 'MEG2212', 'MEG2213',	'MEG2223', 'MEG2222', 'MEG2233', 'MEG2232',	'MEG2242',
		'MEG2243', 'MEG2312', 'MEG2313', 'MEG2323',	'MEG2322', 'MEG2332', 'MEG2333', 'MEG2343',	'MEG2342',
		'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422',	'MEG2433', 'MEG2432', 'MEG2442', 'MEG2443',	'MEG2512',
		'MEG2513', 'MEG2522', 'MEG2523', 'MEG2533',	'MEG2532', 'MEG2543', 'MEG2542', 'MEG2612',	'MEG2613',
		'MEG2623', 'MEG2622', 'MEG2633', 'MEG2632',	'MEG2642', 'MEG2643']
elif channel is 'all_ant':
	ListSensors = ['MEG0113', 'MEG0112', 'MEG0122', 'MEG0123', 'MEG0132', 'MEG0133', 'MEG0143', 'MEG0142',
		'MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0232', 'MEG0233', 'MEG0243', 'MEG0242', 'MEG0313',
		'MEG0312', 'MEG0322', 'MEG0323', 'MEG0333',	'MEG0332', 'MEG0343', 'MEG0342', 'MEG0413',	'MEG0412',
		'MEG0422', 'MEG0423', 'MEG0432', 'MEG0433',	'MEG0443', 'MEG0442', 'MEG0513', 'MEG0512',	'MEG0523',
		'MEG0522', 'MEG0532', 'MEG0533', 'MEG0542',	'MEG0543', 'MEG0613', 'MEG0612', 'MEG0622',	'MEG0623',
		'MEG0633', 'MEG0632', 'MEG0642', 'MEG0643',	'MEG0813', 'MEG0812', 'MEG0822', 'MEG0823',	'MEG0913',
		'MEG0912', 'MEG0923', 'MEG0922', 'MEG0932',	'MEG0933', 'MEG0942', 'MEG0943', 'MEG1013',	'MEG1012',
		'MEG1023', 'MEG1022', 'MEG1032', 'MEG1033',	'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113',	'MEG1123',
		'MEG1122', 'MEG1133', 'MEG1132', 'MEG1142,'	'MEG1143', 'MEG1213', 'MEG1212', 'MEG1223',	'MEG1222',
		'MEG1232', 'MEG1233', 'MEG1243', 'MEG1242',	'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322',	'MEG1333',
		'MEG1332', 'MEG1342', 'MEG1343', 'MEG1412',	'MEG1413', 'MEG1423', 'MEG1422', 'MEG1433',	'MEG1432',	
		'MEG1442', 'MEG1443']

#Path
path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016'

###Load circular-linear correlation results 
for anali, analysis in enumerate(ListAnalysis):

	dat_path = path + '/Data/' + ListFolder[anali] + '/' + ListAnalysis[anali] + '.mat' 
	res_path = path + '/' + ListFolder[anali] + '/Stats'

	print('load: ' + analysis)
	tmp = sio.loadmat(dat_path) #load entire matlab structure resulting from permutation
	raw_scores = np.array(tmp['rho'])
	perm_scores = np.array(tmp['rho_permM'])
	time = np.array(tmp['params']['toi'][0][0])
	del tmp

	tmp = sio.loadmat(path + '/Scripts/SensorClassification.mat')
	sensors = tmp['All2'][0]
	del tmp

	#Convert to readable list of MEG str
	sensors = [sensors[i][0].astype(str) for i, ii in enumerate(sensors)]

	#Find indices of specific channel groups to use
	chan_ind = [sensors.index(chani) for chani in ListSensors]

	#Substract empirical baseline, select groups of sensors to run stats on, & average across these sensors
	all_scores = raw_scores - perm_scores
	all_scores = all_scores[chan_ind, :, :]
	all_scores = np.mean(all_scores, axis=0)
	scores_diag = np.swapaxes(all_scores, 1, 0) #subs x timepoints

	np.save(res_path + '/' + analysis + '_' + channel + '-time.npy', time)
	np.save(res_path + '/' + analysis + '_' + channel + '-all_scores.npy', scores_diag)

	if stat_params is 'permutation':
		#Compute stats against theoretical chance level as obtained by permutations
		print('computing stats based on permutation: ' + analysis)
		p_values_diag = myStats(np.array(scores_diag)[:, :, None] - chance, tail=tail)

	elif stat_params is 'Wilcoxon':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		p_values_diag = parallel_stats(np.array(scores_diag) - chance, correction=False) 

	elif stat_params is 'Wilcoxon-FDR':
		#Compute stats using uncorrected Wilcoxon signed-rank test
		print('computing stats based on corrected Wilcoxon: ' + analysis)
		p_values_diag = parallel_stats(np.array(scores_diag) - chance, correction='FDR') 

	#Save
	np.save(res_path + '/' + analysis + '_' + stat_params + str(tail) + '_' + channel + '-p_values_diag.npy', p_values_diag)