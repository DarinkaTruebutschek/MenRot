#Purpose: This script prepares the data for linear regression based on target location
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 31 January 2017

def menRot_updated_decLoc(wkdir, Condition, Subject, FileName, decCond):
	
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
	from menRot_updated_doRegression_final import menRot_updated_doRegression_final
	from menRot_updated_loc_cfg import (result_path)
	from menRot_updated_loc_cfg import grouped, n_folds
	
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	
	####################################################################
	#Subfunction

	def menRot_prepReg_loc(wkdir, Condition, Subject, FileName, decCond):

		####################################################################
		#Define important variables
		#Import parameters from configuration file
		from menRot_updated_loc_cfg import (data_path, baseline, baseline_loc, downsampling, trainTime, testTime, prediction_method, probabilities, scorer, featureSelection, loc, n_folds, baselineCorr, grouped)
		
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
		else:
			mode = 'mean-prediction'
		
		print(mne.__version__)
		print(mode)
		
		params = {'baseline': baseline, 'baseline_loc': baseline_loc, 'downsampling': downsampling,
		'classification': decCond, 'trainingTime': trainTime, 'testingTime': testTime, 'trainset': trainset, 'testset': testset,
		'prediction_method': prediction_method, 'probabilities': probabilities, 'scorer': scorer, 'featureSelection': featureSelection,
		'mode': mode, 'n_folds': n_folds, 'baselineCorr': baselineCorr}
	
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
		#fname = data_path + '/' + Subject + '_7SD_sss_rot_forDec2.mat' #filtered @30Hz and downsampled to 125Hz
		fname = data_path + '/' + Subject + '_7SD_sss_rot_forDec.mat' #filtered @30Hz
	
		epoch = fldtrp2mne(fname, 'data')
		epoch.info['lowpass'] = float(30) #to correct lowpass filter info
		
		#Downsample data if needed
		if downsampling > 0:
			epoch.decimate(downsampling)
			if loc is 1:
				epoch_loc.decimate(downsampling)
		
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
				
				sel2 = info['seen?'] == 0 #all unseen trials
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
			elif testset is 'Test_AllSeen':
				sel1 = info['seen?'] == 0 #all unseen trials
				
				X_train = epoch[sel1] #select epochs for training
				y_train = np.ravel(np.array(list(info[sel1][label])))
				
				sel2 = info['seen?'] == 1 #all seen trials
				
				X_test = epoch[sel2]
				y_test = np.ravel(np.array(list(info[sel2][label])))
				
		
		print(np.shape(X_train))
		print(np.shape(y_train))
		
		print(np.shape(X_test))
		print(np.shape(y_test))
		
		####################################################################
		#Decoding
		gat, score, diagonal = menRot_updated_doRegression_final(X_train, y_train, X_test, y_test, params)	
					
		return params, epoch.times, gat, score, diagonal #remember to change time to main task and/or localizer task
	
	########################################################################
	#Main part
	params, time, gat, score, diagonal = menRot_prepReg_loc(wkdir, Condition, Subject, FileName, decCond)
	
	if (grouped and n_folds == 5):
		myFolder = '/Grouped/'
	elif (grouped and n_folds == 3):
		myFolder = '/Grouped/ThreeFolds/'
	else:
		myFolder = '/'
		
	#Save data
	fname = result_path + myFolder + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-params'
	np.save(fname, params)
	
	fname1 = result_path + myFolder + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-time'
	np.save(fname1, time)
	
	fname2 = result_path + myFolder + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-gat'
	np.save(fname2, gat)
	
	fname3 = result_path + myFolder + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-score'
	np.save(fname3, score)
	
	fname4 = result_path + myFolder + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-diagonal'
	np.save(fname4, diagonal)
	
	#fname5 = result_path + '/Loc-All/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '-time-loc'
	#np.save(fname5, time_loc)



