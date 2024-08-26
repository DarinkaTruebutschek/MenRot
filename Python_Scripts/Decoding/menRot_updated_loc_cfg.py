#Purpose: This script contains all of the parameters needed for the different
#steps of the analysis. It has to be imported at the beginning of the analysis.
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 08 Nov 2016

########################################################################
#Paths
wkdir = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/'
job_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/SomaWF/Decoding'
data_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Data/mat_artifactRemoved'
result_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding'
script_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/Decoding'

########################################################################
#Preprocessing
baseline_loc = (-0.1, 0.0) #time for the baseline period
baseline = (-0.2, 0.0) #time for baseline period on rotation task
downsampling = 2 #downsampling factor (input at 250Hz)

########################################################################
#Decoding
#decCond = ['loc'] 
trainTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
testTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
#trainTime = {'start': -0.096, 'stop': 0.8, 'step': 0.008, 'length': 0.008}
#testTime = {'start': -0.096, 'stop': 0.8, 'step': 0.008, 'length': 0.008}
#trainTime = {'start': -0.096, 'stop': 0.8, 'step': 0.008, 'length': 0.008}
#testTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
prediction_method = 'predict' #other options: 'predict', 'predict_proba'
probabilities = False
scorer = 'scorer_angle' #other options: 'prob_accuracy', 'scorer_angle'
featureSelection = 1 #indicates how many of the best features to select, if 1 no feature selection
loc = 0 #indicates whether localizer will be used to train the data
n_folds = 3 #normally, this will be a 5-fold cv scheme, but when only looking at certain subsets (e.g., Train_noRot_Test_noRot) it is 2 due to the lower number of trials)
baselineCorr = True
grouped = True #determine whether we are looking at all 24 locations or at 

