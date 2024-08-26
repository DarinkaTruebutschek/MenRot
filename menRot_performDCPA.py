#Purpose: This script computes dPCA as a function of condition of interest
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 4 April 2018

import sys

import mne
import pandas as pd
import pickle

import numpy as np
import scipy.io as sio

from dPCA import dPCA

sys.path.append('/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/')

from fldtrp2mne import fldtrp2mne
from menRot_configDPCA import (ListSubjects,sel_chan, data_path, result_path, baseline, baseline_loc, downsampling, baselineCorr)

#######
for subi, subject in enumerate(ListSubjects):

	#Load data
	fname = data_path + '/' + subject + '_7SD_sss_rot_forDec.mat' #filtered @30Hz
	epoch = fldtrp2mne(fname, 'data')
	epoch.info['lowpass'] = float(30) #to correct lowpass filter info
		
	#Downsample data if needed
	if downsampling > 0:
		epoch.decimate(downsampling)
		
	if baselineCorr:		
		epoch.apply_baseline(baseline)
	
	#Load trialinfo
	mat = sio.loadmat(fname)
	trialinfo = mat['data']['trialInfo']
	condition, behavior = trialinfo[0][0][0][0]
	target = condition[0][0][0].T
	cortarPos = condition[0][0][1].T
	cue = condition[0][0][2].T
	vis = behavior[0][0][0].T
	respnPos = behavior[0][0][1][0][0][0].T
	dis = behavior[0][0][1][0][0][1].T
	dis2Tar = behavior[0][0][1][0][0][2].T
	normDis = behavior[0][0][1][0][0][3].T
	normDis2Tar = behavior[0][0][1][0][0][4].T
	inferTar = behavior[0][0][1][0][0][5].T
	
	task = np.copy(cue)
	task[task != 2] = 1 #Rot
	task[task == 2] = 0 #no Rot
	
	tar_present = np.copy(target)
	tar_present[tar_present != 0] = 1 #target present
	tar_present[tar_present == 0] = 0 #target absent
	
	seen = np.copy(vis)
	seen[seen == 1] = 0 #unseen
	seen[seen > 1] = 1 #seen
		
	#Orientation
	angles = np.deg2rad(np.arange(7.5, 360, 15))
	order = np.array([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
	angles = angles[order]
	target_angles = np.array([angles[t - 1] if t != 0 else np.nan for t in target])
	response_angles = np.array([angles[int(r) - 1] if ~np.isnan(r) else np.nan for r in respnPos])
	inferred_angles = np.array([angles[int(i) - 1] if ~np.isnan(i) else np.nan for i in inferTar])
	
	info = pd.DataFrame(data = np.concatenate((cue, task, tar_present, target, cortarPos, vis, seen, respnPos, dis, dis2Tar, normDis, normDis2Tar), axis = 1),
			columns = ['cue', 'task', 'tarPres', 'tarPos', 'cortarPos', 'vis', 'seen?', 'respnPos', 'dis', 'dis2Tar', 'normDis', 'normDis2Tar'])
	info['angle'] = np.ravel(target_angles)
	info['respAngle'] = np.ravel(response_angles)
	info['inferAngle'] = np.ravel(inferred_angles)

	#First, remove any trials containing nan values in the vis or location response & remove target-absent trials
	correct = (~np.isnan(info['seen?'])) & (~np.isnan(info['respnPos'])) & (info['tarPres'] == 1)
	
	epoch = epoch[correct]
	info = info[correct]

	#Then, select only the subset of time of interest
	epoch.crop(-0.2, None)

	#Third, select the subset of channels to be included
	channels = epoch.info['ch_names']
	if chan_sel is 'mag':
		epoch._pick_types(meg='mag')
	else:
		epoch._pick_types(meg='grad')		

	#Do dPCA
	dataX = epoch.get_data() #n_trials, n_channels, n_time
	print(str(np.shape(dataX)))

	#Select trials
	sel_seen = np.ravel(np.where(info['seen?'] == 1))
	sel_unseen = np.ravel(np.where(info['seen?'] == 0))

	data_seen = dataX[sel_seen, :, :]
	data_unseen = dataX[sel_unseen, :, :]/