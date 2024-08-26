#Purpose: This script prepares the data for linear regression based on target location
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 31 January 2017

def menRot_newPipeline_subscore_decLoc_Riemann(wkdir, Condition, Subject, FileName, decCond):
	
	####################################################################
	#Test input
	#wkdir = 'neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/'
	#Condition = ['Train_All', 'Test_All']
	#Subject = 'lm130479'
	#FileName = 'Test'
	#decCond = 'loc'
	
	####################################################################
	#Load necessary libraries
	import mne
	import os
	
	import numpy as np
	import pandas as pd
	import scipy.io as sio
	
	from scipy import stats
	
	from fldtrp2mne import fldtrp2mne
	from menRot_newPipeline_doRegressionRiemann_subscore import menRot_newPipeline_doRegressionRiemann_subscore
	from menRot_newPipeline_loc_cfg_Riemann import (result_path)
	from menRot_newPipeline_loc_cfg_Riemann import grouped, n_folds
	
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	
	####################################################################
	#Subfunction

	def menRot_prepReg_loc(wkdir, Condition, Subject, FileName, decCond):

		####################################################################
		#Define important variables
		#Import parameters from configuration file
		from menRot_newPipeline_loc_cfg_Riemann import (data_path, baseline, baseline_loc, downsampling, trainTime, testTime, prediction_method, probabilities, scorer, featureSelection, loc, n_folds, baselineCorr, frequency)
		
		#Decoding
		#trainset = Condition[0]
		#testset = Condition[1]
		
		trainset = Condition[0]
		testset = Condition[1]
		
		if Condition[0] is 'Train_loc' and Condition[1] is 'Test_loc': #test on all conditions
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_All' and Condition[1] is 'Test_All':
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_Rot' and Condition[1] is 'Test_Rot':
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_noRot' and Condition[1] is 'Test_noRot':
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_AllSeen' and Condition[1] is 'Test_AllSeen':
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_AllUnseen' and Condition[1] is 'Test_AllUnseen':
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_AllUnseenCorr' and Condition[1] is 'Test_AllUnseenCorr':
			mode = 'cross-validation' 
		elif Condition[0] is 'Train_AllUnseenIncorr' and Condition[1] is 'Test_AllUnseenIncorr':
			mode = 'cross-validation' 
		else:
			mode = 'mean-prediction'
			
		print(mne.__version__)
		print(mode)
		
		params = {'baseline': baseline, 'baseline_loc': baseline_loc, 'downsampling': downsampling,
		'classification': decCond, 'trainingTime': trainTime, 'testingTime': testTime, 'trainset': trainset, 'testset': testset,
		'prediction_method': prediction_method, 'probabilities': probabilities, 'scorer': scorer, 'featureSelection': featureSelection,
		'mode': mode, 'n_folds': n_folds, 'baselineCorr': baselineCorr, 'freq': frequency[0]}
	
		####################################################################
		#Load localizer & apply baseline correction
		if loc is 1:
			fname_loc = data_path + '/' + Subject + '_7SD_sss_loc_forDec.mat' #filtered @30Hz
		
			epoch_loc = fldtrp2mne(fname_loc, 'data')
			epoch_loc.info['lowpass'] = float(30) #to correct lowpass filter info
			
			if baselineCorr:
				epoch_loc.apply_baseline(baseline_loc)
		
			#Load trialinfo for localizer
			mat_loc = sio.loadmat(fname_loc)
			trialinfo_loc = mat_loc['data']['trialInfo']
			condition_loc = trialinfo_loc[0][0][0][0]
			target_loc = condition_loc[0][0][0][0].T
			
			#Orientation for localizer
			if not grouped:
				angles_loc = np.deg2rad(np.arange(7.5, 360, 15))
				order_loc = np.array([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
				angles_loc = angles_loc[order_loc]
				target_angles_loc = np.array([angles_loc[t - 1] if t != 0 else np.nan for t in target_loc]) #there shouldn't be any nans, but I'll leave it just in case
		
				info_loc = pd.DataFrame(data = target_loc, columns = ['tarPos'])
				info_loc['angle'] = np.ravel(target_angles_loc)
			elif grouped:
				#Recode target_loc
				target_loc[target_loc == 2] = 1
				target_loc[target_loc == 3] = 2
				target_loc[target_loc == 4] = 2
				target_loc[target_loc == 5] = 3
				target_loc[target_loc == 6] = 3
				target_loc[target_loc == 7] = 4
				target_loc[target_loc == 8] = 4
				target_loc[target_loc == 9] = 5
				target_loc[target_loc == 10] = 5
				target_loc[target_loc == 11] = 6
				target_loc[target_loc == 12] = 6
				target_loc[target_loc == 13] = 7
				target_loc[target_loc == 14] = 7
				target_loc[target_loc == 15] = 8
				target_loc[target_loc == 16] = 8
				target_loc[target_loc == 17] = 9
				target_loc[target_loc == 18] = 9
				target_loc[target_loc == 19] = 10
				target_loc[target_loc == 20] = 10
				target_loc[target_loc == 21] = 11
				target_loc[target_loc == 22] = 11
				target_loc[target_loc == 23] = 12
				target_loc[target_loc == 24] = 12
				
				angles_loc = np.deg2rad(np.arange(15, 360, 30))
				order_loc = np.array([9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8])
				angles_loc = angles_loc[order_loc]	
				target_angles_loc = np.array([angles_loc[t - 1] if t != 0 else np.nan for t in target_loc]) 
				
				info_loc = pd.DataFrame(data = target_loc, columns = ['tarPos'])
				info_loc['angle'] = np.ravel(target_angles_loc)
		
		####################################################################
		#Load data & apply baseline correction
		#fname = data_path + '/' + Subject + '_7SD_sss_rot_forDec.mat' #filtered @30Hz
	
		#epoch = fldtrp2mne(fname, 'data')
		#epoch.info['lowpass'] = float(30) #to correct lowpass filter info
		
		if frequency[0] is 'all':
			fname = data_path + '/' + Subject + '_7SD_sss_rot_forDec.mat' #filtered @30Hz
	
			epoch = fldtrp2mne(fname, 'data')
			epoch.info['lowpass'] = float(30) #to correct lowpass filter info
	
			#Load trialinfo
			mat = sio.loadmat(fname)
			trialinfo = mat['data']['trialInfo']
			
			#Downsample data if needed
			if downsampling > 0:
				epoch.decimate(downsampling)
			if loc is 1:
				epoch_loc.decimate(downsampling)
		
			if baselineCorr:		
				epoch.apply_baseline(baseline)
		else: 
			fname = data_path + '/' + Subject + '_TFA_' + frequency[0] + '_forDecoding.mat'  #baseline corrected 

			epoch = fldtrp2mne(fname, 'freqDec')
			epoch.info['lowpass'] = float(50) #to correct lowpass filter info
	
			#Load trialinfo
			mat = sio.loadmat(fname)
			trialinfo = mat['freqDec']['trialInfo']
		
		#Load trialinfo
		#mat = sio.loadmat(fname)
		#trialinfo = mat['data']['trialInfo']
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
		if not grouped:
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
		elif grouped:
			#Recode target_loc
			target[target == 2] = 1
			target[target == 3] = 2
			target[target == 4] = 2
			target[target == 5] = 3
			target[target == 6] = 3
			target[target == 7] = 4
			target[target == 8] = 4
			target[target == 9] = 5
			target[target == 10] = 5
			target[target == 11] = 6
			target[target == 12] = 6
			target[target == 13] = 7
			target[target == 14] = 7
			target[target == 15] = 8
			target[target == 16] = 8
			target[target == 17] = 9
			target[target == 18] = 9
			target[target == 19] = 10
			target[target == 20] = 10
			target[target == 21] = 11
			target[target == 22] = 11
			target[target == 23] = 12
			target[target == 24] = 12
			
			#Recode resp
			respnPos[respnPos == 2] = 1
			respnPos[respnPos == 3] = 2
			respnPos[respnPos == 4] = 2
			respnPos[respnPos == 5] = 3
			respnPos[respnPos == 6] = 3
			respnPos[respnPos == 7] = 4
			respnPos[respnPos == 8] = 4
			respnPos[respnPos == 9] = 5
			respnPos[respnPos == 10] = 5
			respnPos[respnPos == 11] = 6
			respnPos[respnPos == 12] = 6
			respnPos[respnPos == 13] = 7
			respnPos[respnPos == 14] = 7
			respnPos[respnPos == 15] = 8
			respnPos[respnPos == 16] = 8
			respnPos[respnPos == 17] = 9
			respnPos[respnPos == 18] = 9
			respnPos[respnPos == 19] = 10
			respnPos[respnPos == 20] = 10
			respnPos[respnPos == 21] = 11
			respnPos[respnPos == 22] = 11
			respnPos[respnPos == 23] = 12
			respnPos[respnPos == 24] = 12
			
			#Recode inferTar
			inferTar[inferTar == 2] = 1
			inferTar[inferTar == 3] = 2
			inferTar[inferTar == 4] = 2
			inferTar[inferTar == 5] = 3
			inferTar[inferTar == 6] = 3
			inferTar[inferTar == 7] = 4
			inferTar[inferTar == 8] = 4
			inferTar[inferTar == 9] = 5
			inferTar[inferTar == 10] = 5
			inferTar[inferTar == 11] = 6
			inferTar[inferTar == 12] = 6
			inferTar[inferTar == 13] = 7
			inferTar[inferTar == 14] = 7
			inferTar[inferTar == 15] = 8
			inferTar[inferTar == 16] = 8
			inferTar[inferTar == 17] = 9
			inferTar[inferTar == 18] = 9
			inferTar[inferTar == 19] = 10
			inferTar[inferTar == 20] = 10
			inferTar[inferTar == 21] = 11
			inferTar[inferTar == 22] = 11
			inferTar[inferTar == 23] = 12
			inferTar[inferTar == 24] = 12
			
			angles = np.deg2rad(np.arange(15, 360, 30))
			order = np.array([9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8])
			angles = angles[order]	
			
			target_angles = np.array([angles[t - 1] if t != 0 else np.nan for t in target])
			response_angles = np.array([angles[int(r) - 1] if ~np.isnan(r) else np.nan for r in respnPos])
			inferred_angles = np.array([angles[int(i) - 1] if ~np.isnan(i) else np.nan for i in inferTar])
	
			info = pd.DataFrame(data = np.concatenate((cue, task, tar_present, target, cortarPos, vis, seen, respnPos, dis, dis2Tar, normDis, normDis2Tar), axis = 1),
			columns = ['cue', 'task', 'tarPres', 'tarPos', 'cortarPos', 'vis', 'seen?', 'respnPos', 'dis', 'dis2Tar', 'normDis', 'normDis2Tar'])
			info['angle'] = np.ravel(target_angles)
			info['respAngle'] = np.ravel(response_angles)
			info['inferAngle'] = np.ravel(inferred_angles)
	
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
		
		#Determine trials used for decoding as well as label
		if decCond is 'loc':
			label = 'angle'
			label_loc = 'angle'
		elif decCond is 'resp':
			label = 'respAngle'
			label_loc = 'angle'
		elif decCond is 'infer':
			label = 'inferAngle'
			label_loc = 'angle'
			
		if trainset is 'Train_loc':
			X_train = epoch_loc #select epochs for training
			y_train = np.array(info_loc[label_loc]) #select trial info
			if testset is 'Test_rot':
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_loc': 
				X_test = epoch_loc #select epochs for testing
				y_test = np.array(info_loc[label_loc]) #select trial info
			elif testset is 'Test_noRot': 
				sel1 = info['task'] == 0 #select only no rotation trials
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Rot': 
				sel2 = info['task'] == 1 #select only rotation trials
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Left': 
				sel1 = info['cue'] == 1 #select only left rotation trials
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Right': 
				sel2 = info['cue'] == 3 #select only right rotation trials
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_noRot_Seen': 
				sel1 = info['task'] == 0 #no Rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_noRot_Unseen': 
				sel1 = info['task'] == 0 #no Rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Rot_Seen': 
				sel1 = info['task'] == 1 #rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Rot_Unseen': 
				sel1 = info['task'] == 1 #rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Left_Seen': 
				sel1 = info['cue'] == 1 #left
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Left_Unseen': 
				sel1 = info['cue'] == 1 #left
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Right_Seen': 
				sel1 = info['cue'] == 3 #right
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 1 #seen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_Right_Unseen': 
				sel1 = info['cue'] == 3 #right
				epoch = epoch[sel1]
				info = info[sel1]
				
				sel2 = info['seen?'] == 0 #unseen
				epoch = epoch[sel2]
				info = info[sel2]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_AllSeen': 
				sel1 = info['seen?'] == 1 #seen
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
			elif testset is 'Test_AllUnseen': 
				sel1 = info['seen?'] == 0 #unseen
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_test = epoch #select epochs for testing
				y_test = np.ravel(np.array(list(info[label])))
		elif trainset is 'Train_All': #decoder trained on all available data
			if testset is 'Test_All':
				X_train = epoch #select epochs for training
				y_train = np.ravel(np.array(list(info[label])))
				
				#All vis
				sel_seen = np.ravel(np.where(np.array(info['seen?'] == 1))) 
				sel_unseen = np.ravel(np.where(np.array(info['seen?'] == 0)))	
							
				sel_corr = (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_corr = np.ravel(np.where(sel_corr))		
						
				sel_incorr = (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_incorr = np.ravel(np.where(sel_incorr))
				
				#All rot
				sel_rot = (info['task'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot = np.ravel(np.where(sel_rot))
				
				sel_noRot = (info['task'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot = np.ravel(np.where(sel_noRot))
				
				#Rot
				sel_rot_seen = (info['task'] == 1) & (info['seen?'] == 1) 
				sel_rot_seen = np.ravel(np.where(sel_rot_seen))
				
				sel_rot_unseen = (info['task'] == 1) & (info['seen?'] == 0) 
				sel_rot_unseen = np.ravel(np.where(sel_rot_unseen))
				
				sel_rot_seen_corr = (info['task'] == 1) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot_seen_corr = np.ravel(np.where(sel_rot_seen_corr))
				
				sel_rot_corr = (info['task'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot_corr = np.ravel(np.where(sel_rot_corr))		
						
				sel_rot_incorr = (info['task'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_rot_incorr = np.ravel(np.where(sel_rot_incorr))
				
				#No rot
				sel_noRot_seen = (info['task'] == 0) & (info['seen?'] == 1) 
				sel_noRot_seen = np.ravel(np.where(sel_noRot_seen))
				
				sel_noRot_unseen = (info['task'] == 0) & (info['seen?'] == 0) 
				sel_noRot_unseen = np.ravel(np.where(sel_noRot_unseen))
				
				sel_noRot_seen_corr = (info['task'] == 0) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot_seen_corr = np.ravel(np.where(sel_noRot_seen_corr))
				
				sel_noRot_corr = (info['task'] == 0) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot_corr = np.ravel(np.where(sel_noRot_corr))		
						
				sel_noRot_incorr = (info['task'] == 0) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_noRot_incorr = np.ravel(np.where(sel_noRot_incorr))
				
				#Left 
				sel_left_seen = (info['cue'] == 3) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_left_seen = np.ravel(np.where(sel_left_seen))
				
				sel_left_unseen = (info['cue'] == 3) & (info['seen?'] == 0) 
				sel_left_unseen = np.ravel(np.where(sel_left_unseen))
				
				sel_left_corr = (info['cue'] == 3) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_left_corr = np.ravel(np.where(sel_left_corr))		
						
				sel_left_incorr = (info['cue'] == 3) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_left_incorr = np.ravel(np.where(sel_left_incorr))
				
				#Right 
				sel_right_seen = (info['cue'] == 1) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_right_seen = np.ravel(np.where(sel_right_seen))
				
				sel_right_unseen = (info['cue'] == 1) & (info['seen?'] == 0) 
				sel_right_unseen = np.ravel(np.where(sel_right_unseen))
				
				sel_right_corr = (info['cue'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_right_corr = np.ravel(np.where(sel_right_corr))		
						
				sel_right_incorr = (info['cue'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_right_incorr = np.ravel(np.where(sel_right_incorr))
				
				X_test = epoch
				y_test = np.ravel(np.array(list(info[label])))
		elif trainset is 'Train_Rot': #decoder trained on all available data
			if testset is 'Test_Rot':
				sel1 = info['task'] == 1 #rot
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_train = epoch #select epochs for training
				y_train = np.ravel(np.array(list(info[label])))
			
				X_test = epoch
				y_test = np.ravel(np.array(list(info[label])))
				
								#All vis
				sel_seen = np.ravel(np.where(np.array(info['seen?'] == 1))) 
				sel_unseen = np.ravel(np.where(np.array(info['seen?'] == 0)))	
							
				sel_corr = (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_corr = np.ravel(np.where(sel_corr))		
						
				sel_incorr = (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_incorr = np.ravel(np.where(sel_incorr))
				
				#All rot
				sel_rot = (info['task'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot = np.ravel(np.where(sel_rot))
				
				sel_noRot = (info['task'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot = np.ravel(np.where(sel_noRot))
				
				#Rot
				sel_rot_seen = (info['task'] == 1) & (info['seen?'] == 1) 
				sel_rot_seen = np.ravel(np.where(sel_rot_seen))
				
				sel_rot_unseen = (info['task'] == 1) & (info['seen?'] == 0) 
				sel_rot_unseen = np.ravel(np.where(sel_rot_unseen))
				
				sel_rot_seen_corr = (info['task'] == 1) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot_seen_corr = np.ravel(np.where(sel_rot_seen_corr))
				
				sel_rot_corr = (info['task'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot_corr = np.ravel(np.where(sel_rot_corr))		
						
				sel_rot_incorr = (info['task'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_rot_incorr = np.ravel(np.where(sel_rot_incorr))
				
				#No rot
				sel_noRot_seen = (info['task'] == 0) & (info['seen?'] == 1) 
				sel_noRot_seen = np.ravel(np.where(sel_noRot_seen))
				
				sel_noRot_unseen = (info['task'] == 0) & (info['seen?'] == 0) 
				sel_noRot_unseen = np.ravel(np.where(sel_noRot_unseen))
				
				sel_noRot_seen_corr = (info['task'] == 0) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot_seen_corr = np.ravel(np.where(sel_noRot_seen_corr))
				
				sel_noRot_corr = (info['task'] == 0) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot_corr = np.ravel(np.where(sel_noRot_corr))		
						
				sel_noRot_incorr = (info['task'] == 0) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_noRot_incorr = np.ravel(np.where(sel_noRot_incorr))
				
				#Left 
				sel_left_seen = (info['cue'] == 3) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_left_seen = np.ravel(np.where(sel_left_seen))
				
				sel_left_unseen = (info['cue'] == 3) & (info['seen?'] == 0) 
				sel_left_unseen = np.ravel(np.where(sel_left_unseen))
				
				sel_left_corr = (info['cue'] == 3) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_left_corr = np.ravel(np.where(sel_left_corr))		
						
				sel_left_incorr = (info['cue'] == 3) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_left_incorr = np.ravel(np.where(sel_left_incorr))
				
				#Right 
				sel_right_seen = (info['cue'] == 1) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_right_seen = np.ravel(np.where(sel_right_seen))
				
				sel_right_unseen = (info['cue'] == 1) & (info['seen?'] == 0) 
				sel_right_unseen = np.ravel(np.where(sel_right_unseen))
				
				sel_right_corr = (info['cue'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_right_corr = np.ravel(np.where(sel_right_corr))		
						
				sel_right_incorr = (info['cue'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_right_incorr = np.ravel(np.where(sel_right_incorr))
				
			elif testset is 'Test_noRot':
				sel1 = info['task'] == 1 #rot
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				
				sel2 = info['task'] == 0 # no rot
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
		elif trainset is 'Train_noRot': #decoder trained on all available data
			if testset is 'Test_noRot':
				sel1 = info['task'] == 0 #noRot
				epoch = epoch[sel1]
				info = info[sel1]
				
				X_train = epoch #select epochs for training
				y_train = np.ravel(np.array(list(info[label])))
			
				X_test = epoch
				y_test = np.ravel(np.array(list(info[label])))
				
				#All vis
				sel_seen = np.ravel(np.where(np.array(info['seen?'] == 1))) 
				sel_unseen = np.ravel(np.where(np.array(info['seen?'] == 0)))	
							
				sel_corr = (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_corr = np.ravel(np.where(sel_corr))		
						
				sel_incorr = (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_incorr = np.ravel(np.where(sel_incorr))
				
				#All rot
				sel_rot = (info['task'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot = np.ravel(np.where(sel_rot))
				
				sel_noRot = (info['task'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot = np.ravel(np.where(sel_noRot))
				
				#Rot
				sel_rot_seen = (info['task'] == 1) & (info['seen?'] == 1) 
				sel_rot_seen = np.ravel(np.where(sel_rot_seen))
				
				sel_rot_unseen = (info['task'] == 1) & (info['seen?'] == 0) 
				sel_rot_unseen = np.ravel(np.where(sel_rot_unseen))
				
				sel_rot_seen_corr = (info['task'] == 1) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot_seen_corr = np.ravel(np.where(sel_rot_seen_corr))
				
				sel_rot_corr = (info['task'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_rot_corr = np.ravel(np.where(sel_rot_corr))		
						
				sel_rot_incorr = (info['task'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_rot_incorr = np.ravel(np.where(sel_rot_incorr))
				
				#No rot
				sel_noRot_seen = (info['task'] == 0) & (info['seen?'] == 1) 
				sel_noRot_seen = np.ravel(np.where(sel_noRot_seen))
				
				sel_noRot_unseen = (info['task'] == 0) & (info['seen?'] == 0) 
				sel_noRot_unseen = np.ravel(np.where(sel_noRot_unseen))
				
				sel_noRot_seen_corr = (info['task'] == 0) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot_seen_corr = np.ravel(np.where(sel_noRot_seen_corr))
				
				sel_noRot_corr = (info['task'] == 0) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_noRot_corr = np.ravel(np.where(sel_noRot_corr))		
						
				sel_noRot_incorr = (info['task'] == 0) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_noRot_incorr = np.ravel(np.where(sel_noRot_incorr))
				
				#Left 
				sel_left_seen = (info['cue'] == 3) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_left_seen = np.ravel(np.where(sel_left_seen))
				
				sel_left_unseen = (info['cue'] == 3) & (info['seen?'] == 0) 
				sel_left_unseen = np.ravel(np.where(sel_left_unseen))
				
				sel_left_corr = (info['cue'] == 3) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_left_corr = np.ravel(np.where(sel_left_corr))		
						
				sel_left_incorr = (info['cue'] == 3) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_left_incorr = np.ravel(np.where(sel_left_incorr))
				
				#Right 
				sel_right_seen = (info['cue'] == 1) & (info['seen?'] == 1) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_right_seen = np.ravel(np.where(sel_right_seen))
				
				sel_right_unseen = (info['cue'] == 1) & (info['seen?'] == 0) 
				sel_right_unseen = np.ravel(np.where(sel_right_unseen))
				
				sel_right_corr = (info['cue'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)
				sel_right_corr = np.ravel(np.where(sel_right_corr))		
						
				sel_right_incorr = (info['cue'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))
				sel_right_incorr = np.ravel(np.where(sel_right_incorr))
				
			elif testset is 'Test_Rot':
				sel1 = info['task'] == 0 #no rot
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				
				sel2 = info['task'] == 1 #rot
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
		elif trainset is 'Train_AllSeen':
			if testset is 'Test_AllSeen':
				sel1 = info['seen?'] == 1 #all seen trials
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				
				sel2 = info['seen?'] == 1 #all seen trials
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
			elif testset is 'Test_AllUnseen':
				sel1 = info['seen?'] == 1 #all seen trials
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				
				sel2 = info['seen?'] == 0 #all unseen trials
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
		elif trainset is 'Train_AllUnseen':
			if testset is 'Test_AllUnseen':
				sel1 = info['seen?'] == 0 #all unseen trials
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				#y_train = np.random.shuffle(y_train) #just to see whether there is also this baseline offset when shuffling the labels
				
				sel2 = info['seen?'] == 0 #all unseen trials
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
				#y_test = y_train #just to see whether there is also this baseline offset when shuffling the labels
			elif testset is 'Test_AllSeen':
				sel1 = info['seen?'] == 0 #all unseen trials
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				
				sel2 = info['seen?'] == 1 #all seen trials
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
		elif trainset is 'Train_AllUnseenCorr':
			if testset is 'Test_AllUnseenCorr':
				sel1 = info['seen?'] == 0 #all unseen trials
				sel2 = (info['dis'] >= -2) & (info['dis'] <= 2) #all correct trials
				
				sel3 = sel1 & sel2
				
				X_train = epoch[sel3] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel3][label])))

				X_test = epoch[sel3]
				y_test = np.ravel(np.array(list(info[sel3][label])))
		elif trainset is 'Train_AllUnseenIncorr':
			if testset is 'Test_AllUnseenIncorr':
				sel1 = info['seen?'] == 0 #all unseen trials
				sel2 = (info['dis'] >= -2) & (info['dis'] <= 2) #all correct trials
				sel3 = sel1 & sel2
				
				X_train = epoch[sel3] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel3][label])))

				X_test = epoch[sel3]
				y_test = np.ravel(np.array(list(info[sel3][label])))			

		print(np.shape(X_train))
		print(np.shape(y_train))
		
		print(np.shape(X_test))
		print(np.shape(y_test))
		
		####################################################################
		#Decoding
		(score_seen, score_unseen, score_corr, score_incorr, 
		score_rot_seen, score_rot_unseen, score_rot_corr, score_rot_incorr, 
		score_noRot_seen, score_noRot_unseen, score_noRot_corr, score_noRot_incorr,
		score_left_seen, score_left_unseen, score_left_corr, score_left_incorr, 
		score_right_seen, score_right_unseen, score_right_corr, score_right_incorr, 
		score_rot, score_noRot, score_rot_seen_corr, score_noRot_seen_corr,
		label_seen, label_unseen, label_corr, label_incorr, 
		label_rot_seen, label_rot_unseen, label_rot_corr, label_rot_incorr, 
		label_noRot_seen, label_noRot_unseen, label_noRot_corr, label_noRot_incorr,
		label_left_seen, label_left_unseen, label_left_corr, label_left_incorr, 
		label_right_seen, label_right_unseen, label_right_corr, label_right_incorr,
		label_rot, label_noRot, label_rot_seen_corr, label_noRot_seen_corr) = menRot_newPipeline_doRegressionRiemann_subscore(X_train, y_train, X_test, y_test, params, 
		sel_seen, sel_unseen, sel_corr, sel_incorr, 
		sel_rot_seen, sel_rot_unseen, sel_rot_corr, sel_rot_incorr, 
		sel_noRot_seen, sel_noRot_unseen, sel_noRot_corr, sel_noRot_incorr, 
		sel_left_seen, sel_left_unseen, sel_left_corr, sel_left_incorr, 
		sel_right_seen, sel_right_unseen, sel_right_corr, sel_right_incorr,
		sel_rot, sel_noRot, sel_rot_seen_corr, sel_noRot_seen_corr, result_path, FileName, Subject, Condition)	
					
		return (params, epoch.times,
		score_seen, score_unseen, score_corr, score_incorr, 
		score_rot_seen, score_rot_unseen, score_rot_corr, score_rot_incorr, 
		score_noRot_seen, score_noRot_unseen, score_noRot_corr, score_noRot_incorr,
		score_left_seen, score_left_unseen, score_left_corr, score_left_incorr, 
		score_right_seen, score_right_unseen, score_right_corr, score_right_incorr, 
		score_rot, score_noRot, score_rot_seen_corr, score_noRot_seen_corr,
		label_seen, label_unseen, label_corr, label_incorr, 
		label_rot_seen, label_rot_unseen, label_rot_corr, label_rot_incorr, 
		label_noRot_seen, label_noRot_unseen, label_noRot_corr, label_noRot_incorr,
		label_left_seen, label_left_unseen, label_left_corr, label_left_incorr, 
		label_right_seen, label_right_unseen, label_right_corr, label_right_incorr,
		label_rot, label_noRot, label_rot_seen_corr, label_noRot_seen_corr, frequency)
		 #remember to change time to main task and/or localizer task
	
	########################################################################
	#Main part
	(params, time, 
	score_seen, score_unseen, score_corr, score_incorr, 
	score_rot_seen, score_rot_unseen, score_rot_corr, score_rot_incorr, 
	score_noRot_seen, score_noRot_unseen, score_noRot_corr, score_noRot_incorr,
	score_left_seen, score_left_unseen, score_left_corr, score_left_incorr, 
	score_right_seen, score_right_unseen, score_right_corr, score_right_incorr, 
	score_rot, score_noRot, score_rot_seen_corr, score_noRot_seen_corr,
	label_seen, label_unseen, label_corr, label_incorr, 
	label_rot_seen, label_rot_unseen, label_rot_corr, label_rot_incorr, 
	label_noRot_seen, label_noRot_unseen, label_noRot_corr, label_noRot_incorr,
	label_left_seen, label_left_unseen, label_left_corr, label_left_incorr, 
	label_right_seen, label_right_unseen, label_right_corr, label_right_incorr,
	label_rot, label_noRot, label_rot_seen_corr, label_noRot_seen_corr, frequency) = menRot_prepReg_loc(wkdir, Condition, Subject, FileName, decCond)
	
	#if (grouped and n_folds == 5):
		#myFolder = '/Grouped/FiveFolds/Subscore/TFA/'
	#elif (grouped and n_folds == 4):
		#myFolder = '/Grouped/FourFolds/'
	#elif (grouped and n_folds == 8):
		#myFolder = '/Grouped/EightFolds/Subscore/'
	#elif (grouped and n_folds == 2):
		#myFolder = '/Grouped/TwoFolds/Subscore/'
	#else:
		#myFolder = '/'
	
	#Save data
	fname = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-params'
	np.save(fname, params)
	
	fname1 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' +frequency[0] + '-time'
	np.save(fname1, time)
	
	#fname2 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] +'-gat'
	#np.save(fname2, gat)
	
	fname3 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreSeen'
	np.save(fname3, score_seen)
	
	fname4 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreUnseen'
	np.save(fname4, score_unseen)
	
	fname5 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreUnseenCorr'
	np.save(fname5, score_corr)
	
	fname6 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreUnseenIncorr'
	np.save(fname6, score_incorr)
	
	
	fname7 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRotSeen'
	np.save(fname7, score_rot_seen)
	
	fname8 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRotUnseen'
	np.save(fname8, score_rot_unseen)
	
	fname9 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRotUnseenCorr'
	np.save(fname9, score_rot_corr)
	
	fname10 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRotUnseenIncorr'
	np.save(fname10, score_rot_incorr)
	
	
	fname11 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreNoRotSeen'
	np.save(fname11, score_noRot_seen)
	
	fname12 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreNoRotUnseen'
	np.save(fname12, score_noRot_unseen)
	
	fname13 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreNoRotUnseenCorr'
	np.save(fname13, score_noRot_corr)
	
	fname14 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreNoRotUnseenIncorr'
	np.save(fname14, score_noRot_incorr)
	
	
	fname15 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreLeftSeen'
	np.save(fname15, score_left_seen)
	
	fname16 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreLeftUnseen'
	np.save(fname16, score_left_unseen)
	
	fname17 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreLeftUnseenCorr'
	np.save(fname17, score_left_corr)
	
	fname18 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreLeftUnseenIncorr'
	np.save(fname18, score_left_incorr)
	
	
	fname19 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRightSeen'
	np.save(fname19, score_right_seen)
	
	fname20 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRightUnseen'
	np.save(fname20, score_right_unseen)
	
	fname21 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRightUnseenCorr'
	np.save(fname21, score_right_corr)
	
	fname22 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRightUnseenIncorr'
	np.save(fname22, score_right_incorr)
	
	
	fname23 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelSeen'
	np.save(fname23, label_seen)
	
	fname24 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelUnseen'
	np.save(fname24, label_unseen)
	
	fname25 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelUnseenCorr'
	np.save(fname25, label_corr)
	
	fname26 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelUnseenIncorr'
	np.save(fname26, label_incorr)
	
	
	fname27 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRotSeen'
	np.save(fname27, label_rot_seen)
	
	fname28 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRotUnseen'
	np.save(fname28, label_rot_unseen)
	
	fname29 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRotUnseenCorr'
	np.save(fname29, label_rot_corr)
	
	fname30 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRotUnseenIncorr'
	np.save(fname30, label_rot_incorr)
	
	
	fname31 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelNoRotSeen'
	np.save(fname31, label_noRot_seen)
	
	fname32 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelNoRotUnseen'
	np.save(fname32, label_noRot_unseen)
	
	fname33 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelNoRotUnseenCorr'
	np.save(fname33, label_noRot_corr)
	
	fname34 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelNoRotUnseenIncorr'
	np.save(fname34, label_noRot_incorr)
	
	
	fname35 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelLeftSeen'
	np.save(fname35, label_left_seen)
	
	fname36 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelLeftUnseen'
	np.save(fname36, label_left_unseen)
	
	fname37 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelLeftUnseenCorr'
	np.save(fname37, label_left_corr)
	
	fname38 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelLeftUnseenIncorr'
	np.save(fname38, label_left_incorr)
	
	
	
	fname39 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRightSeen'
	np.save(fname39, label_right_seen)
	
	fname40 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRightUnseen'
	np.save(fname40, label_right_unseen)
	
	fname41 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRightUnseenCorr'
	np.save(fname41, label_right_corr)
	
	fname42 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRightUnseenIncorr'
	np.save(fname42, label_right_incorr)
	
	
	fname43 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRot'
	np.save(fname43, label_rot)
	
	fname44 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelNoRot'
	np.save(fname44, label_noRot)
	
	fname45 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-labelRotSeenCorr'
	np.save(fname45, label_rot_seen_corr)
	
	fname46 = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] +  '_' + frequency[0] + '-labelNoRotSeenCorr'
	np.save(fname46, label_noRot_seen_corr)
	
	
	fname47 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRot'
	np.save(fname47, score_rot)
	
	fname48 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreNoRot'
	np.save(fname48, score_noRot)
	
	fname49= result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreRotSeenCorr'
	np.save(fname49, score_rot_seen_corr)
	
	fname50 = result_path + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-scoreNoRotSeenCorr'
	np.save(fname50, score_noRot_seen_corr)
	
	#fname5 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-time-loc'
	#np.save(fname5, time_loc)



