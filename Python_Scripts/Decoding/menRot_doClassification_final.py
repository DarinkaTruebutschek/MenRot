#Purpose: This function runs a simple two-class classification. 
#It takes as input a data matrix(X) and a label vector(y). 
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 06 Nov 2016

###Load libraries###
import sys

#Basics
import numpy as np
import pickle

#MNE
from mne.decoding import GeneralizationAcrossTime

#Sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

#JR tools
from jr.gat.scorers import prob_accuracy, scorer_auc, scorer_angle

def menRot_doClassification_final(X_train, y_train, X_test, y_test, params, result_path, Condition, Subject):
	"This function performs a simple two-class classification."
	
	#Initialize output variable
	score = []
	
	####################################################################	
	#Learning machinery
	#Scaler
	scaler = StandardScaler()
	
	#Feature selection
	if params['featureSelection'] > 1:
		fs = SelectKBest(f_classif, k = params['featureSelection'])
	
	#Model
	model = svm.SVC(C = 1, kernel = 'linear', class_weight = 'balanced', probability = params['probabilities'], decision_function_shape = 'ovr')
	
	#Pipeline
	if params['featureSelection'] > 1:
		clf = make_pipeline(fs, scaler, model)
	else:
		clf = make_pipeline(scaler, model)
		
	#Cross-validation
	cv = StratifiedKFold(y_train, 5)
	
	#Identify prediction mode & prediction method
	predict_mode = params['mode']
	predict_method = params['prediction_method']
	
	#Identify scorer
	if params['scorer'] is 'scorer_auc':
		scorer = scorer_auc
	elif params['scorer'] is 'prob_accuracy':
		scorer = prob_accuracy
	elif params['scorer'] is 'scorer_angle':
		scorer = prob_accuracy
	
	####################################################################
	#Learning process
	gat = GeneralizationAcrossTime(clf = clf, cv = cv, train_times = params['trainingTime'], test_times = params['testingTime'], predict_mode = predict_mode, predict_method = predict_method, scorer = scorer, n_jobs = 8)
	
	gat.fit(X_train, y = y_train)
	
	#Save gat as pickle
	fsave = result_path + '/' + Condition[0] + '_' + Condition[1] + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + params['freq'] + '-gatForWeights.p'
	pickle.dump(gat, open(fsave, "wb"))
	
	score = gat.score(X_test, y = y_test)
	#gat.predict(X_test) #This results in a matrix (train_times, test_times, trials, predicted angle, predicted radius)
	#scores = gat.score()
	
	#gat.n_jobs = 1 #for some reason this is required as parallelization does not seem to work ...
	#Subscore seen/unseen
	#score_rot = subscore(gat, sel_rot, y = y_test[sel_rot])
	#score_rot = np.array(score_rot)
	
	#score_noRot = subscore(gat, sel_unseen, y = y_test[sel_unseen])
	#score_unseen = np.array(score_unseen)
	
	score = np.array(score)
	diagonal = np.diagonal(score)
	
	return gat, score, diagonal

