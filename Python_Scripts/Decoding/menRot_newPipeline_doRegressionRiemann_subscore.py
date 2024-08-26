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
from mne.decoding import Vectorizer
from mne.decoding import UnsupervisedSpatialFilter
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
from sklearn.decomposition import PCA

#Pyriemann
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import (ERPCovariances, XdawnCovariances, HankelCovariances)

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

def menRot_newPipeline_doRegressionRiemann_subscore(X_train, y_train, X_test, y_test, params, 
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

	#Pipeline
	if params['featureSelection'] > 1:
		clf = make_pipeline(fs, scaler, UnsupervisedSpatialFilter(PCA(70), average=False),
		ERPCovariances(estimator='lwf'),
		TangentSpace('logeuclid'),
		PolarRegression(Ridge()))
	else:
		clf = make_pipeline(UnsupervisedSpatialFilter(PCA(70), average=False),
		ERPCovariances(estimator='lwf'),
		TangentSpace('logeuclid'),
		PolarRegression(Ridge()))
		
	#Cross-validation
	cv = StratifiedKFold(params['n_folds'])

	#Identify scorer
	scorer = scorer_angle
	
	####################################################################
	#Prep input data
	X_train = X_train.get_data()
	X_test = X_test.get_data()
	
	time = np.linspace(-0.5, 3.5, 500)
	time_bins = [[37, 63], [75, 100], [100,138], [138, 284], [284, 471], [471, 500], [284, 296], [296, 308], [308, 320], [320, 332], [332, 344], [344, 356], [356, 368], [368, 380], [380, 392], [392, 404], [404, 416], [416, 428], [428, 440], [440, 458], [458, 470]]
	
	preds = np.zeros((X_test.shape[0], 2, len(time_bins))) #n_trials * n_predictions(position and radius)
	
	score_seen = np.zeros(len(time_bins))
	score_unseen = np.zeros(len(time_bins))
	score_corr = np.zeros(len(time_bins))
	score_incorr = np.zeros(len(time_bins))
	
	score_rot_seen = np.zeros(len(time_bins))
	score_rot_unseen = np.zeros(len(time_bins))
	score_rot_corr = np.zeros(len(time_bins))
	score_rot_incorr = np.zeros(len(time_bins))
	
	score_noRot_seen = np.zeros(len(time_bins))
	score_noRot_unseen = np.zeros(len(time_bins))
	score_noRot_corr = np.zeros(len(time_bins))
	score_noRot_incorr = np.zeros(len(time_bins))
	
	score_left_seen = np.zeros(len(time_bins))
	score_left_unseen = np.zeros(len(time_bins))
	score_left_corr = np.zeros(len(time_bins))
	score_left_incorr = np.zeros(len(time_bins))	
	
	score_right_seen = np.zeros(len(time_bins))
	score_right_unseen = np.zeros(len(time_bins))
	score_right_corr = np.zeros(len(time_bins))
	score_right_incorr = np.zeros(len(time_bins))	
	
	score_rot =  np.zeros(len(time_bins))	
	score_noRot =  np.zeros(len(time_bins))	
	
	score_rot_seen_corr =  np.zeros(len(time_bins))	
	score_noRot_seen_corr =  np.zeros(len(time_bins))	
	####################################################################
	#Learning process
	for t, timei in enumerate(time_bins):
		print(str(timei))
		Xtr = X_train[:, :, timei[0] : timei[1]]
		Xte = X_test[:, :, timei[0] : timei[1]]
		
		print(np.shape(Xtr))
		print(np.shape(Xte))
		
		for train_index, test_index in cv.split(Xtr, y_train):
			print(train_index)
			print(test_index)
			print('Entering cv loop')
		
			#Fit 
			clf.fit(Xtr[train_index], y_train[train_index])
		
			#Predict
			pr = clf.predict(Xte[test_index])
			preds[test_index, 0, t] += pr[:, 0]
			preds[test_index, 1, t] += pr[:, 1] #n_trials x n_predictions (i.e., position and radius) x n_timeBins

		#Subscore seen/unseen
		score_seen[t] = scorer_angle(y_test[sel_seen], preds[sel_seen, :, t])
		label_seen = np.array(sel_seen)
	
		score_unseen[t] = scorer_angle(y_test[sel_unseen], preds[sel_unseen, :, t])
		label_unseen = np.array(sel_unseen)
	
		score_corr[t] = scorer_angle(y_test[sel_corr], preds[sel_corr, :, t])
		label_corr = np.array(sel_corr)
	
		score_incorr[t] = scorer_angle(y_test[sel_incorr], preds[sel_incorr, :, t])
		label_incorr = np.array(sel_incorr)
	
	
		score_rot_seen[t] = scorer_angle(y_test[sel_rot_seen], preds[sel_rot_seen, :, t])
		label_rot_seen = np.array(sel_rot_seen)
	
		score_rot_unseen[t] = scorer_angle(y_test[sel_rot_unseen], preds[sel_rot_unseen, :, t])
		label_rot_unseen = np.array(sel_rot_unseen)
	
		score_rot_corr[t] = scorer_angle(y_test[sel_rot_corr], preds[sel_rot_corr, :, t])
		label_rot_corr = np.array(sel_rot_corr)
	
		score_rot_incorr[t] = scorer_angle(y_test[sel_rot_incorr], preds[sel_rot_incorr, :, t])
		label_rot_incorr = np.array(sel_rot_incorr)
	
	
		score_noRot_seen[t] = scorer_angle(y_test[sel_noRot_seen], preds[sel_noRot_seen, :, t])
		label_noRot_seen = np.array(sel_noRot_seen)
	
		score_noRot_unseen[t] = scorer_angle(y_test[sel_noRot_unseen], preds[sel_noRot_unseen, :, t])
		label_noRot_unseen = np.array(sel_noRot_unseen)
	
		score_noRot_corr[t] = scorer_angle(y_test[sel_noRot_corr], preds[sel_noRot_corr, :, t])
		label_noRot_corr = np.array(sel_noRot_corr)
	
		score_noRot_incorr[t] = scorer_angle(y_test[sel_noRot_incorr], preds[sel_noRot_incorr, :, t])
		label_noRot_incorr = np.array(sel_noRot_incorr)
	
	
		score_left_seen[t] = scorer_angle(y_test[sel_left_seen], preds[sel_left_seen, :, t])
		label_left_seen = np.array(sel_left_seen)
	
		score_left_unseen[t] = scorer_angle(y_test[sel_left_unseen], preds[sel_left_unseen, :, t])
		label_left_unseen = np.array(sel_left_unseen)
	
		score_left_corr[t] = scorer_angle(y_test[sel_left_corr], preds[sel_left_corr, :, t])
		label_left_corr = np.array(sel_left_corr)
	
		score_left_incorr[t] = scorer_angle(y_test[sel_left_incorr], preds[sel_left_incorr, :, t])
		label_left_incorr = np.array(sel_left_incorr)
	
	
		score_right_seen[t] = scorer_angle(y_test[sel_right_seen], preds[sel_right_seen, :, t])
		label_right_seen = np.array(sel_right_seen)
	
		score_right_unseen[t] = scorer_angle(y_test[sel_right_unseen], preds[sel_right_unseen, :, t])
		label_right_unseen = np.array(sel_right_unseen)
	
		score_right_corr[t] = scorer_angle(y_test[sel_right_corr], preds[sel_right_corr, :, t])
		label_right_corr = np.array(sel_right_corr)
	
		score_right_incorr[t] = scorer_angle(y_test[sel_right_incorr], preds[sel_right_incorr, :, t])
		label_right_incorr = np.array(sel_right_incorr)
	
	
		score_rot[t] = scorer_angle(y_test[sel_rot], preds[sel_rot, :, t])
		label_rot = np.array(sel_rot)
	
		score_noRot[t] = scorer_angle(y_test[sel_noRot], preds[sel_noRot, :, t])
		label_noRot = np.array(sel_noRot)
	
		score_rot_seen_corr[t] = scorer_angle(y_test[sel_rot_seen_corr], preds[sel_rot_seen_corr, :, t])
		label_rot_seen_corr = np.array(sel_rot_seen_corr)
	
		score_noRot_seen_corr[t] = scorer_angle(y_test[sel_noRot_seen_corr], preds[sel_noRot_seen_corr, :, t])
		label_noRot_seen_corr = np.array(sel_noRot_seen_corr)
	
	#Save as pickle
	fsave = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] + '_' + params['freq'] + '-y_pred.p'
	pickle.dump(preds, open(fsave, "wb"))

	fsave = result_path  + FileName + '/IndRes/' + Subject + '_' + Condition[0] + '_' + Condition[1] +  '_' + params['freq'] + '-gatForDis.p'
	pickle.dump(clf, open(fsave, "wb"))
	
	return (score_seen, score_unseen, score_corr, score_incorr, 
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
