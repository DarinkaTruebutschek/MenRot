#Purpose: This script prepares the data for linear regression based on target location
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 31 January 2017

def menRot_decLoc(wkdir, Condition, Subject):
	
	####################################################################
	#Test input
	#wkdir = 'neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/'
	#Condition = ['Train_loc', 'Test_rot']
	#Subject = 'lm130479'
	
	####################################################################
	#Load necessary libraries
	import mne
	import os
	
	import numpy as np
	import pandas as pd
	import scipy.io as sio
	
	from scipy import stats
	
	from fldtrp2mne import fldtrp2mne
	from menRot_doRegression_final import menRot_doRegression_final
	from menRot_loc_cfg import (result_path)
	
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	
	####################################################################
	#Subfunction

	def menRot_prepReg_loc(wkdir, Condition, Subject):

		####################################################################
		#Define important variables
		#Import parameters from configuration file
		from menRot_loc_cfg import (data_path, baseline, baseline_loc, downsampling, decCond, trainTime, testTime, prediction_method, probabilities, scorer, featureSelection, loc)
		
		#Decoding
		trainset = Condition[0]
		testset = Condition[1]
		
		if Condition[1] is 'Test_loc': #test on all conditions
			mode = 'cross-validation' 
		elif Condition[1] is 'Test_All':
			mode = 'cross-validation' 
		elif Condition[1] is 'Test_Rot':
			mode = 'cross-validation' 
		elif Condition[1] is 'Test_noRot':
			mode = 'cross-validation' 
		else:
			mode = 'mean-prediction'
	
		params = {'baseline': baseline, 'baseline_loc': baseline_loc, 'downsampling': downsampling,
		'classification': decCond, 'trainingTime': trainTime, 'testingTime': testTime, 'trainset': trainset, 'testset': testset,
		'prediction_method': prediction_method, 'probabilities': probabilities, 'scorer': scorer, 'featureSelection': featureSelection,
		'mode': mode}
	
		####################################################################
		#Load localizer & apply baseline correction
		if loc is 1:
			fname_loc = data_path + '/' + Subject + '_7SD_sss_loc_forDec2.mat' #filtered @30Hz
		
			epoch_loc = fldtrp2mne(fname_loc, 'data')
			epoch_loc.info['lowpass'] = float(30) #to correct lowpass filter info
			epoch_loc.apply_baseline(baseline_loc)
		
			#Load trialinfo for localizer
			mat_loc = sio.loadmat(fname_loc)
			trialinfo_loc = mat_loc['data']['trialInfo']
			condition_loc = trialinfo_loc[0][0][0][0]
			target_loc = condition_loc[0][0][0][0].T
		
			#Orientation for localizer
			angles_loc = np.deg2rad(np.arange(7.5, 360, 15))
			order_loc = np.array([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
			angles_loc = angles_loc[order_loc]
			target_angles_loc = np.array([angles_loc[t - 1] if t != 0 else np.nan for t in target_loc]) #there shouldn't be any nans, but I'll leave it just in case
		
			info_loc = pd.DataFrame(data = target_loc, columns = ['tarPos'])
			info_loc['angle'] = np.ravel(target_angles_loc)
		
		####################################################################
		#Load data & apply baseline correction
		fname = data_path + '/' + Subject + '_7SD_sss_rot_forDec2.mat' #filtered @30Hz
	
		epoch = fldtrp2mne(fname, 'data')
		epoch.info['lowpass'] = float(30) #to correct lowpass filter info
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
	
		info = pd.DataFrame(data = np.concatenate((cue, task, tar_present, target, cortarPos, vis, seen, respnPos, dis, dis2Tar, normDis, normDis2Tar), axis = 1),
		columns = ['cue', 'task', 'tarPres', 'tarPos', 'cortarPos', 'vis', 'seen?', 'respnPos', 'dis', 'dis2Tar', 'normDis', 'normDis2Tar'])
		info['angle'] = np.ravel(target_angles)
	
		#Downsample data if needed
		if downsampling > 0:
			epoch.decimate(downsampling)
			if loc is 1:
				epoch_loc.decimate(downsampling)
	
		#First, remove any trials containing nan values in the vis response
		correct = ~np.isnan(info['seen?'])
	
		epoch = epoch[correct]
		info = info[correct]
		
		#Second, remove any trials containing nan values in location response
		correct2 = ~np.isnan(info['respnPos'])
		
		epoch = epoch[correct2]
		info = info[correct2]
		
		#Third, remove any target-absent trials
		correct3 = info['tarPres'] == 1
		
		epoch = epoch[correct3]
		info = info[correct3]
		
		if trainset is 'Train_loc':
			X_train = epoch_loc #select epochs for training
			y_train = np.array(info_loc['angle']) #select trial info
			if testset is 'Test_rot':
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_loc': 
				X_test = epoch_loc #select epochs for testing
				y_test = np.array(info_loc['angle']) #select trial info
			elif testset is 'Test_noRot': 
				sel1 = info['task'] == 0 #select only no rotation trials
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Rot': 
				sel2 = info['task'] == 1 #select only rotation trials
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Left': 
				sel1 = info['cue'] == 1 #select only left rotation trials
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Right': 
				sel2 = info['cue'] == 3 #select only right rotation trials
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_noRot_Seen': 
				sel1 = info['task'] == 0 #no Rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_noRot_Unseen': 
				sel1 = info['task'] == 0 #no Rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Rot_Seen': 
				sel1 = info['task'] == 1 #rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Rot_Unseen': 
				sel1 = info['task'] == 1 #rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Left_Seen': 
				sel1 = info['cue'] == 1 #left
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Left_Unseen': 
				sel1 = info['cue'] == 1 #left
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Right_Seen': 
				sel1 = info['cue'] == 3 #right
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Right_Unseen': 
				sel1 = info['cue'] == 3 #right
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Seen': 
				sel1 = info['seen?'] == 1 #seen
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
			elif testset is 'Test_Unseen': 
				sel1 = info['seen?'] == 0 #unseen
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info['angle'])))
		elif trainset is 'Train_All': #decoder trained on all available data
			X_train = epoch #select epochs for training
			y_train = np.ravel(np.array(list(info['angle'])))
			
			X_test = epoch
			y_test = np.ravel(np.array(list(info['angle'])))
		elif trainset is 'Train_Rot': #decoder trained on all available data
			if testset is 'Test_Rot':
				sel1 = info['task'] == 1 #rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_train = epoch #select epochs for training
				y_train = np.ravel(np.array(list(info['angle'])))
			
				X_test = epoch
				y_test = np.ravel(np.array(list(info['angle'])))
		elif trainset is 'Train_noRot': #decoder trained on all available data
			if testset is 'Test_noRot':
				sel1 = info['task'] == 0 #noRot
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_train = epoch #select epochs for training
				y_train = np.ravel(np.array(list(info['angle'])))
			
				X_test = epoch
				y_test = np.ravel(np.array(list(info['angle'])))
		
		####################################################################
		#Decoding
		gat, score, diagonal = menRot_doRegression_final(X_train, y_train, X_test, y_test, params)	
					
		return params, epoch.times, gat, score, diagonal
	
	########################################################################
	#Main part
	params, time, gat, score, diagonal = menRot_prepReg_loc(wkdir, Condition, Subject)
	
	#Save data
	fname = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-params'
	np.save(fname, params)
	
	fname1 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-time'
	np.save(fname1, time)
	
	fname2 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-gat'
	np.save(fname2, gat)
	
	fname3 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-score'
	np.save(fname3, score)
	
	fname4 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-diagonal'
	np.save(fname4, diagonal)
	
	#fname5 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-time-loc'
	#np.save(fname5, time_loc)



