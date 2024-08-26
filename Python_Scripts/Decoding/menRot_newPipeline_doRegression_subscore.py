#Purpose: This function runs a multivariate regression analysis. 
#It takes as input a data matrix(X) and a label vector(y). 
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 31 January 2017

###Load libraries###
import sys

#Basics
import numpy as np
import pickle

#MNE
from mne.decoding import GeneralizationAcrossTime, GeneralizingEstimator

#Sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge

#JR tools
from jr.gat.base import subscore, subselect_ypred
from jr.gat.scorers import prob_accuracy, scorer_auc
from jr.gat.scorers import scorer_angle #as _scorer_angle
from jr.stats import corr_linear_circular
from jr.gat.classifiers import SVR_angle, PolarRegression
#from jr.gat.classifiers import SVR_angle, AngularRegression

#def scorer_angle(y_true, y_pred):
	#y_pred = np.array(y_pred)
	#if y_pred.ndim == 1:
		#y_pred = y_pred[:, np.newaxis]
	#return _scorer_angle(y_true, y_pred[:, 0])

def menRot_newPipeline_doRegression_subscore(X_train, y_train, X_test, y_test, params, 
sel_seen, sel_unseen, sel_corr, sel_incorr, 
sel_rot_seen, sel_rot_unseen, sel_rot_corr, sel_rot_incorr, 
sel_noRot_seen, sel_noRot_unseen, sel_noRot_corr, sel_noRot_incorr, 
sel_left_seen, sel_left_unseen, sel_left_corr, sel_left_incorr, 
sel_right_seen, sel_right_unseen, sel_right_corr, sel_right_incorr,
sel_rot, sel_noRot, sel_rot_seen_corr, sel_noRot_seen_corr,
result_path, FileName, Subject, Condition):
	"This function performs a multivariate regression."
	
	#Initialize output variable
	score = []
	
	####################################################################	
	#Learning machinery
	#Scaler
	scaler = StandardScaler()
	
	#Feature selection
	if params['featureSelection'] > 1:
		fs = SelectKBest(f_regression, k = params['featureSelection']) #changed to correct feature selection; used to be f_classif
	
	#Model
	#model = AngularRegression() #directly calls SVM.LinearSVR with C=1
	model = PolarRegression(Ridge())
	
	#Pipeline
	if params['featureSelection'] > 1:
		clf = make_pipeline(fs, scaler, model)
	else:
		clf = make_pipeline(scaler, model)
		
	#Cross-validation
	#cv = StratifiedKFold(y_train, 5)
	cv = StratifiedKFold(params['n_folds'])
	
	#Identify prediction mode & prediction method
	predict_mode = params['mode']
	predict_method = params['prediction_method']
	
	#Identify scorer
	scorer = scorer_angle
	
	####################################################################
	#Prep input data
	#X_train = X_train.get_data()
	#X_test = X_test.get_data()
	
	####################################################################
	#Learning process
	gat = GeneralizationAcrossTime(clf = clf, cv = cv, train_times = params['trainingTime'], test_times = params['testingTime'], predict_mode = predict_mode, predict_method = predict_method, scorer = scorer, n_jobs = 8)
	#gat = GeneralizingEstimator(clf, scoring = scorer, n_jobs = 8)
	
	gat.fit(X_train, y = y_train)
	
	#Save gat as pickle
	fsave = result_path + '/newPipeline/Grouped/FiveFolds/Subscore/FeatureSelection/' + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + params['freq'] + '-gatForWeights.p'
	pickle.dump(gat, open(fsave, "wb"))
	
	gat.predict(X_test) #This results in a matrix (train_times, test_times, trials, predicted angle, predicted radius)
	y_pred = gat.y_pred_
	fsave = result_path + '/newPipeline/Grouped/FiveFolds/Subscore/FeatureSelection/' + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + params['freq'] + '-y_pred.p'
	pickle.dump(y_pred, open(fsave, "wb"))
	#scores = gat.score()
	
	#Save gat as pickle
	fsave = result_path + '/newPipeline/Grouped/FiveFolds/Subscore/FeatureSelection/' + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + params['freq'] + '-gatForDis.p'
	pickle.dump(gat, open(fsave, "wb"))
	
	#gat.n_jobs = 1 #for some reason this is required as parallelization does not seem to work ...
	#Subscore seen/unseen
	score_seen = subscore(gat, sel_seen, y = y_test[sel_seen])
	score_seen = np.array(score_seen)
	label_seen = np.array(sel_seen)
	
	score_unseen = subscore(gat, sel_unseen, y = y_test[sel_unseen])
	score_unseen = np.array(score_unseen)
	label_unseen = np.array(sel_unseen)
	
	score_corr = subscore(gat, sel_corr, y = y_test[sel_corr])
	score_corr = np.array(score_corr)
	label_corr = np.array(sel_corr)
	
	score_incorr = subscore(gat, sel_incorr, y = y_test[sel_incorr])
	score_incorr = np.array(score_incorr)
	label_incorr = np.array(sel_incorr)
	
	
	score_rot_seen = subscore(gat, sel_rot_seen, y = y_test[sel_rot_seen])
	score_rot_seen = np.array(score_rot_seen)
	label_rot_seen = np.array(sel_rot_seen)
	
	score_rot_unseen = subscore(gat, sel_rot_unseen, y = y_test[sel_rot_unseen])
	score_rot_unseen = np.array(score_rot_unseen)
	label_rot_unseen = np.array(sel_rot_unseen)
	
	score_rot_corr = subscore(gat, sel_rot_corr, y = y_test[sel_rot_corr])
	score_rot_corr = np.array(score_rot_corr)
	label_rot_corr = np.array(sel_rot_corr)
	
	score_rot_incorr = subscore(gat, sel_rot_incorr, y = y_test[sel_rot_incorr])
	score_rot_incorr = np.array(score_rot_incorr)
	label_rot_incorr = np.array(sel_rot_incorr)
	
	
	score_noRot_seen = subscore(gat, sel_noRot_seen, y = y_test[sel_noRot_seen])
	score_noRot_seen = np.array(score_noRot_seen)
	label_noRot_seen = np.array(sel_noRot_seen)
	
	score_noRot_unseen = subscore(gat, sel_noRot_unseen, y = y_test[sel_noRot_unseen])
	score_noRot_unseen = np.array(score_noRot_unseen)
	label_noRot_unseen = np.array(sel_noRot_unseen)
	
	score_noRot_corr = subscore(gat, sel_noRot_corr, y = y_test[sel_noRot_corr])
	score_noRot_corr = np.array(score_noRot_corr)
	label_noRot_corr = np.array(sel_noRot_corr)
	
	score_noRot_incorr = subscore(gat, sel_noRot_incorr, y = y_test[sel_noRot_incorr])
	score_noRot_incorr = np.array(score_noRot_incorr)
	label_noRot_incorr = np.array(sel_noRot_incorr)
	
	
	score_left_seen = subscore(gat, sel_left_seen, y = y_test[sel_left_seen])
	score_left_seen = np.array(score_left_seen)
	label_left_seen = np.array(sel_left_seen)
	
	score_left_unseen = subscore(gat, sel_left_unseen, y = y_test[sel_left_unseen])
	score_left_unseen = np.array(score_left_unseen)
	label_left_unseen = np.array(sel_left_unseen)
	
	score_left_corr = subscore(gat, sel_left_corr, y = y_test[sel_left_corr])
	score_left_corr = np.array(score_left_corr)
	label_left_corr = np.array(sel_left_corr)
	
	score_left_incorr = subscore(gat, sel_left_incorr, y = y_test[sel_left_incorr])
	score_left_incorr = np.array(score_left_incorr)
	label_left_incorr = np.array(sel_left_incorr)
	
	
	score_right_seen = subscore(gat, sel_right_seen, y = y_test[sel_right_seen])
	score_right_seen = np.array(score_right_seen)
	label_right_seen = np.array(sel_right_seen)
	
	score_right_unseen = subscore(gat, sel_right_unseen, y = y_test[sel_right_unseen])
	score_right_unseen = np.array(score_right_unseen)
	label_right_unseen = np.array(sel_right_unseen)
	
	score_right_corr = subscore(gat, sel_right_corr, y = y_test[sel_right_corr])
	score_right_corr = np.array(score_right_corr)
	label_right_corr = np.array(sel_right_corr)
	
	score_right_incorr = subscore(gat, sel_right_incorr, y = y_test[sel_right_incorr])
	score_right_incorr = np.array(score_right_incorr)
	label_right_incorr = np.array(sel_right_incorr)
	
	
	score_rot = subscore(gat, sel_rot, y = y_test[sel_rot])
	score_rot = np.array(score_rot)
	label_rot = np.array(sel_rot)
	
	score_noRot = subscore(gat, sel_noRot, y = y_test[sel_noRot])
	score_noRot = np.array(score_noRot)
	label_noRot = np.array(sel_noRot)
	
	score_rot_seen_corr = subscore(gat, sel_rot_seen_corr, y = y_test[sel_rot_seen_corr])
	score_rot_seen_corr = np.array(score_rot_seen_corr)
	label_rot_seen_corr = np.array(sel_rot_seen_corr)
	
	score_noRot_seen_corr = subscore(gat, sel_noRot_seen_corr, y = y_test[sel_noRot_seen_corr])
	score_noRot_seen_corr = np.array(score_noRot_seen_corr)
	label_noRot_seen_corr = np.array(sel_noRot_seen_corr)
	
	return (gat, score_seen, score_unseen, score_corr, score_incorr, 
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
	label_rot, label_noRot, label_rot_seen_corr, label_noRot_seen_corr)
