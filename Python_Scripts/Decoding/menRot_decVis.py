#Purpose: This script prepares the data for binary classification 
#based on visibility.
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 08 Nov 2016

def menRot_decVis(wkdir, Condition, Subject):
	
	####################################################################
	#Test input
	#wkdir = '/neurospin/meg/meg_tmp/WMP_Darinka_2015/'
	#Condition = ['Train_P', 'Test_P']
	#Subject = 'ab140235'
	
	####################################################################
	#Load necessary libraries
	import mne
	import os
	
	import numpy as np
	import pandas as pd
	import scipy.io as sio
	
	from scipy import stats
	
	from fldtrp2mne import fldtrp2mne
	from menRot_doClassification_final import menRot_doClassification_final
	from menRot_cfg import (result_path)
	
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	
	####################################################################
	#Subfunction

	def menRot_prepDec_vis(wkdir, Condition, Subject):

		####################################################################
		#Define important variables
		#Import parameters from configuration file
		from menRot_cfg import (data_path, baseline, downsampling, decCond, trainTime, testTime, prediction_method, probabilities, scorer, featureSelection, baselineCorr, frequency)
	
		#Decoding
		trainset = Condition[0]
		testset = Condition[1]
		
		if (decCond[0] is 'seen' and decCond[1] is 'unseen') or decCond[0] is 'present' or decCond[0] is  'unseenCorr':
			if trainset[-4] == testset[-4]: #Check whether we are training and testing on the same data
				mode = 'cross-validation'
			else:
				mode = 'mean-prediction'
		elif (decCond[0] is 'seen' and decCond[1] is 'unseenCorr') or (decCond[0] is 'seen' and decCond[1] is 'unseenIncorr'):
			mode = 'cross-validation'
		else:
			if trainset[5 :] == testset[4 :]:
				mode = 'cross-validation'
			else:
				mode = 'mean-prediction'			
	
		params = {'baseline': baseline, 'downsampling': downsampling,
		'classification': decCond, 'trainingTime': trainTime, 'testingTime': testTime, 'trainset': trainset, 'testset': testset,
		'prediction_method': prediction_method, 'probabilities': probabilities, 'scorer': scorer, 'featureSelection': featureSelection,
		'mode': mode, 'freq': frequency[0]}
	
		####################################################################
		#Load data & apply baseline correction
		print(mne.__version__)
		print(mode)
		
		if frequency is 'all':
			fname = data_path + '/' + Subject + '_7SD_sss_rot_forDec.mat' #filtered @30Hz
	
			epoch = fldtrp2mne(fname, 'data')
			epoch.info['lowpass'] = float(30) #to correct lowpass filter info
	
			#Load trialinfo
			mat = sio.loadmat(fname)
			trialinfo = mat['data']['trialInfo']
		else: 
			fname = data_path + '/' + Subject + '_TFA_' + frequency[0] + '_forDecoding.mat'  #baseline corrected 

			epoch = fldtrp2mne(fname, 'freqDec')
			epoch.info['lowpass'] = float(50) #to correct lowpass filter info
	
			#Load trialinfo
			mat = sio.loadmat(fname)
			trialinfo = mat['freqDec']['trialInfo']
			
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
		
		acc = np.copy(dis)
		acc[(acc >= -2) & (acc <= 2)] = 1 #correct
		acc[acc != 1] = 0 #incorrect
	
		#Orientation
		angles = np.deg2rad(np.arange(7.5, 360, 15))
		order = np.array([18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
		angles = angles[order]
		target_angles = np.array([angles[t - 1] if t != 0 else np.nan for t in target])
	
		info = pd.DataFrame(data = np.concatenate((task, tar_present, target, cortarPos, vis, seen, respnPos, dis, dis2Tar, normDis, normDis2Tar, acc), axis = 1),
		columns = ['task', 'tarPres', 'tarPos', 'cortarPos', 'vis', 'seen?', 'respnPos', 'dis', 'dis2Tar', 'normDis', 'normDis2Tar', 'acc'])
		info['angle'] = np.ravel(target_angles)
	
		#Downsample data if needed
		if downsampling > 0:
			epoch.decimate(downsampling)
		
		if baselineCorr:		
			epoch.apply_baseline(baseline)
	
		#First, remove any trials containing nan values in the vis response
		correct = ~np.isnan(info['seen?'])
	
		epoch = epoch[correct]
		info = info[correct]
		
		#Second, remove any trials containing nan values in location response
		correct2 = ~np.isnan(info['respnPos'])
		
		epoch = epoch[correct2]
		info = info[correct2]
	
		#Check which part of the dataset to use
		if decCond[0] is 'seen':
			present = info['tarPres'] == 1 #select only target-present trials when decoding seen vs. unseen
		
			epoch = epoch[present] #select relevant epochs
			info = info[present] #selecht relevant trial info
		elif decCond[0] is 'unseenCorr':
			present = (info['tarPres'] == 1) & (info['seen?'] == 0) #select only target-present, unseen trials when decoding accuracy in unseen trials
		
			epoch = epoch[present] #select relevant epochs
			info = info[present] #selecht relevant trial info
		elif decCond[0] is 'present':
			epoch = epoch
			info = info
		elif decCond[0] is 'vis': #for this classification, we will train each visibility category vs absent and look at the generalization
			epoch = epoch
			info = info
		
		if trainset is 'Train_noRot' or trainset is 'Train_noRotUnseenAcc':
			if testset is 'Test_noRot' or testset is 'Test_noRotUnseenAcc':
				sel1 = info['task'] == 0 #select only no rot trials
				sel2 = info['task'] == 0
			elif testset is 'Test_Rot':
				sel1 = info['task'] == 0 #select only no rot trials as training set
				sel2 = info['task'] == 1 #select rot trials as test set
		elif trainset is 'Train_Rot' or trainset is 'Train_RotUnseenAcc':
			if testset is 'Test_Rot' or testset is 'Test_RotUnseenAcc':
				sel1 = info['task'] == 1 #select only rot trials
				sel2 = info['task'] == 1
			elif testset is 'Test_noRot':
				sel1 = info['task'] == 1 #select rot trials as training set
				sel2 = info['task'] == 0 #select only no rot trials as test set
		elif trainset is 'Train_All' or trainset is 'Train_AllUnseenAcc' or trainset is 'Train_AllSeen': 
			if testset is 'Test_All' or testset is 'Test_AllUnseenAcc':
				sel1 = info['task'] >= 0
				sel2 = info['task'] >= 0
			elif testset is 'Test_AllUnseenIncorr':
				sel1 = (info['task'] >= 0) & ((info['seen?'] == 1) | ((info['seen?'] == 0) & (info['acc'] == 0)))
				sel2 = (info['task'] >= 0) & ((info['seen?'] == 1) | ((info['seen?'] == 0) & (info['acc'] == 0)))
			elif testset is 'Test_AllUnseenCorr':
				sel1 = (info['task'] >= 0) & ((info['seen?'] == 1) | ((info['seen?'] == 0) & (info['acc'] == 1)))
				sel2 = (info['task'] >= 0) & ((info['seen?'] == 1) | ((info['seen?'] == 0) & (info['acc'] == 1)))
				
		elif trainset is 'Train_SeenAbsent':
			if testset is 'Test_SeenAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 1)) | ((info['tarPres'] == 0) & (info['seen?'] == 0)) #target-present seen & target-absent unseen
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 1)) | ((info['tarPres'] == 0) & (info['seen?'] == 0))
				
				label1 = np.array(info[sel1]['seen?']) #1 = seen, 0 = absent
				label2 = np.array(info[sel2]['seen?']) #1 = seen, 0 = absent
			elif testset is 'Test_UnseenCorrAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 1)) | ((info['tarPres'] == 0) & (info['seen?'] == 0))
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				
				label1 = np.array(info[sel1]['seen?']) #1 = seen, 0 = absent
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 1 #recode unseen correct as 1
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 3 #recode unseen incorrect as 3

				label2 = tmp[(tmp == 1) | (tmp == 0)] 
				
				#Sanity check
				if np.shape(label2) != np.sum(sel2):
					print('Fatal error')
	
			elif testset is 'Test_UnseenIncorrAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 1)) | ((info['tarPres'] == 0) & (info['seen?'] == 0))
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				
				label1 = np.array(info[sel1]['seen?']) #1 = seen, 0 = absent
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 3 #recode unseen correct as 3
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 1 #recode unseen incorrect as 1

				label2 = tmp[(tmp == 1) | (tmp == 0)] 
				
				#Sanity check
				if np.shape(label2) != np.sum(sel2):
					print('Fatal error')
				
		elif trainset is 'Train_UnseenCorrAbsent':
			if testset is 'Test_UnseenCorrAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 1 #recode unseen correct as 1
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 3 #recode unseen incorrect as 3
				
				label1 = tmp[(tmp == 1) | (tmp == 0)] 
				label2 = np.copy(label1)
				
				#Sanity check
				if np.shape(label1) != np.sum(sel1):
					print('Fatal error')
				
			elif testset is 'Test_UnseenIncorrAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 1 #recode unseen correct as 1
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 3 #recode unseen incorrect as 3
				
				label1 = tmp[(tmp == 1) | (tmp == 0)] 
				label2 = tmp[(tmp == 3) | (tmp == 0)] 
				label2[label2 == 3] = 1
				
				#Sanity check
				if (np.shape(label1) != np.sum(sel1)) | (np.shape(label2) != np.sum(sel2)):
					print('Fatal error')
				
			elif testset is 'Test_SeenAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 1)) | ((info['tarPres'] == 0) & (info['seen?'] == 0)) #target-present seen & target-absent unseen
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 1 #recode unseen correct as 1
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 3 #recode unseen incorrect as 3
				
				label1 = tmp[(tmp == 1) | (tmp == 0)] 
				label2 = np.array(info[sel2]['seen?']) #1 = seen, 0 = absent
				
				#Sanity check
				if (np.shape(label1) != np.sum(sel1)) | (np.shape(label2) != np.sum(sel2)):
					print('Fatal error')				
				
		elif trainset is 'Train_UnseenIncorrAbsent':
			if testset is 'Test_UnseenIncorrAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 3 #recode unseen correct as 3
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 1 #recode unseen incorrect as 1
				
				label1 = tmp[(tmp == 1) | (tmp == 0)] 
				label2 = np.copy(label1)
				
				#Sanity check
				if np.shape(label1) != np.sum(sel1):
					print('Fatal error')
					
			elif testset is 'Test_UnseenCorrAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 3 #recode unseen correct as 3
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 1 #recode unseen incorrect as 1
				
				label1 = tmp[(tmp == 1) | (tmp == 0)] 
				label2 = tmp[(tmp == 3) | (tmp == 0)] 
				label2[label2 == 3] = 1
				
				#Sanity check
				if (np.shape(label1) != np.sum(sel1)) | (np.shape(label2) != np.sum(sel2)):
					print('Fatal error')
					
			elif testset is 'Test_SeenAbsent':
				sel1 = ((info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))) | (info['tarPres'] == 0) & (info['seen?'] == 0)
				sel2 = ((info['tarPres'] == 1) & (info['seen?'] == 1)) | ((info['tarPres'] == 0) & (info['seen?'] == 0)) #target-present seen & target-absent unseen
				
				tmp = np.copy(info['seen?'])
				tmp[tmp == 1] = 2 #recode seen as 2
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & (info['dis'] >= -2) & (info['dis'] <= 2)] = 3 #recode unseen correct as 3
				tmp[(info['tarPres'] == 1) & (info['seen?'] == 0) & ((info['dis'] < -2) | (info['dis'] > 2))] = 1 #recode unseen incorrect as 1
				
				label1 = tmp[(tmp == 1) | (tmp == 0)] 
				label2 = np.array(info[sel2]['seen?']) #1 = seen, 0 = absent
				
				#Sanity check
				if (np.shape(label1) != np.sum(sel1)) | (np.shape(label2) != np.sum(sel2)):
					print('Fatal error')	
				
		if decCond[0] is 'seen':
			X_train = epoch[sel1] #select epochs for training
			y_train = np.array(info[sel1]['seen?']) #select trial info
			X_test = epoch[sel2] #select epochs for testing
			y_test = np.array(info[sel2]['seen?']) #select trial info	
		elif decCond[0] is 'unseenCorr':
			X_train = epoch[sel1] #select epochs for training
			y_train = np.array(info[sel1]['acc']) #select trial info
			X_test = epoch[sel2] #select epochs for testing
			y_test = np.array(info[sel2]['acc']) #select trial info	
		elif decCond[0] is 'present':
			X_train = epoch[sel1] #select epochs for training
			y_train = np.array(info[sel1]['tarPres']) #select trial info
			X_test = epoch[sel2] #select epochs for testing
			y_test = np.array(info[sel2]['tarPres']) #select trial info	
		elif decCond[0] is 'vis':
			X_train = epoch[sel1]
			y_train = label1 #select trial info
			X_test = epoch[sel2] #select epochs for testing
			y_test = label2
		
		print(np.shape(X_train))
		print(np.shape(y_train))
		print(np.shape(X_test))
		print(np.shape(y_test))
		
		####################################################################
		#Decoding
		gat, score, diagonal = menRot_doClassification_final(X_train, y_train, X_test, y_test, params, result_path, Condition, Subject)	
					
		return params, epoch.times, gat, score, diagonal, frequency
	
	########################################################################
	#Main part
	params, time, gat, score, diagonal, frequency = menRot_prepDec_vis(wkdir, Condition, Subject)
	
	#Save data
	fname = result_path + '/' + Condition[0] + '_' + Condition[1] + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] +  '_' + frequency[0] + '-params'
	np.save(fname, params)
	
	fname1 = result_path + '/' + Condition[0] + '_' + Condition[1] + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] +'-time'
	np.save(fname1, time)
	
	#fname2 = result_path + '/' + Condition[0] + '_' + Condition[1] + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-gat'
	#np.save(fname2, gat)
	
	fname3 = result_path + '/' + Condition[0] + '_' + Condition[1] + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-score'
	np.save(fname3, score)
	
	fname4 = result_path + '/' + Condition[0] + '_' + Condition[1] + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + frequency[0] + '-diagonal'
	np.save(fname4, diagonal)



