#Purpose: This script contains all of the parameters needed for the dPCA 
#analysis. It has to be imported at the beginning of the analysis.
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 4 April 2018

########################################################################
ListSubjects = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477','bo160176', 'at140305', 
	'df150397', 'rl130571', 'mm140137', 'mb160304', 'lk160274', 'av160302', 'cc150418', 
	'cs150204', 'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 
	'cc130066', 'in110286', 'ss120102']
ListSubjects = ['lm130479']

sel_chan = 'mag'

########################################################################
#Paths
wkdir = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/'
data_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Data/mat_artifactRemoved'
result_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/StateSpace/Vis'
script_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/dPCA'

########################################################################
#Preprocessing
baseline_loc = (-0.1, 0.0) #time for the baseline period
baseline = (-0.2, 0.0) #time for baseline period on rotation task
downsampling = 2 #downsampling factor (input at 250Hz)

########################################################################
#Decoding
decCond = ['Vis'] 
trainTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
testTime = {'start': -0.2, 'stop': 3.496, 'step': 0.008, 'length': 0.008}
n_folds = 5 #normally, this will be a 5-fold cv scheme, but when only looking at certain subsets (e.g., Train_noRot_Test_noRot) it is 2 due to the lower number of trials)
baselineCorr = True


