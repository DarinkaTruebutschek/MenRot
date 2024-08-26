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
result_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding_Vis_TFA'
script_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/Decoding'

########################################################################
#Preprocessing
baseline = (-0.2, 0.0) #time for the baseline period
downsampling = 0 #downsampling factor (input at 250Hz) when doing ERFs, otherwise, nothing

########################################################################
#Decoding
decCond = ['unseenCorr', 'unseenCorr'] #or: ['seen', 'unseenCorr'], ['present', 'absent'] ['vis', 'absent']
trainTime = {'start': -0.2, 'stop': 3.24, 'step': 0.02, 'length': 0.02}
testTime = {'start': -0.2, 'stop': 3.24, 'step': 0.02, 'length': 0.02}
#trainTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
#testTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
prediction_method = 'decision_function' #other options: 'predict', 'predict_proba'
probabilities = True
scorer = 'scorer_auc' #other options: 'prob_accuracy', 'scorer_angle' 'scorer_auc'
featureSelection = 1 #indicates how many of the best features to select, if 1 no feature selection
loc = 0 #indicates whether localizer will be used to train the data
baselineCorr = False #this needs to be set to true when decoding on ERFs, & false when decoding on TFA
frequency = ['HighBeta']



