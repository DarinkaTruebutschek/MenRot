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
#trainTime = {'start': -0.2, 'stop': 3.24, 'step': 0.02, 'length': 0.02} #for timeFrequency
#testTime = {'start': -0.2, 'stop': 3.24, 'step': 0.02, 'length': 0.02}
#trainTime = {'start': -0.2, 'stop': 3.49, 'step': 0.1, 'length': 0.1}
#testTime = {'start': -0.2, 'stop': 3.49, 'step': 0.1, 'length': 0.1}
prediction_method = 'predict' #other options: 'predict', 'predict_proba'
probabilities = False
scorer = 'scorer_angle' #other options: 'prob_accuracy', 'scorer_angle'
featureSelection = 1 #indicates how many of the best features to select, if 1 no feature selection
loc = 0 #indicates whether localizer will be used to train the data
n_folds = 5 #normally, this will be a 5-fold cv scheme, but when only looking at certain subsets (e.g., Train_noRot_Test_noRot) it is 2 due to the lower number of trials)
grouped = True

baselineCorr = True #this needs to be set to true when decoding on ERFs, & false when decoding on TFA
frequency = ['all']
selChannels = ['MEG21', 'MEG17', 'MEG153', 'MEG152', 'MEG19', 'MEG16', 'MEG18', 'MEG074', 'MEG073', 'MEG20', 'MEG23', 'MEG25', 'MEG22', 'MEG24', 'MEG264', 'MEG263']