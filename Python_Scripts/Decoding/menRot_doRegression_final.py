#Purpose: This function runs a multivariate regression analysis. 
#It takes as input a data matrix(X) and a label vector(y). 
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 31 January 2017

###Load libraries###
import sys

#Basics
import numpy as np

#MNE
from mne.decoding import GeneralizationAcrossTime

#Sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

#JR tools
from jr.gat.scorers import prob_accuracy, scorer_auc, scorer_angle
from jr.gat.classifiers import SVR_angle, AngularRegression

def menRot_doRegression_final(X_train, y_train, X_test, y_test, params):
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
	model = AngularRegression() #directly calls SVM.LinearSVR with C=1
	
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
	scorer = scorer_angle
	
	####################################################################
	#Learning process
	gat = GeneralizationAcrossTime(clf = clf, cv = cv, train_times = params['trainingTime'], test_times = params['testingTime'], predict_mode = predict_mode, predict_method = predict_method, scorer = scorer, n_jobs = 8)
	
	gat.fit(X_train, y = y_train)
	score = gat.score(X_test, y = y_test)
	
	score = np.array(score)
	diagonal = np.diagonal(score)
	
	return gat, score, diagonal

