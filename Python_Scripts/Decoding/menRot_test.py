####################Load necessary libraries####################
#%pylab inline

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy import stats
from scipy.stats import wilcoxon

#Add personal functions to python path
sys.path.append('/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/Decoding/')

#from mne.time_frequency.tfr import _check_decim
from jr2.plot import base, gat_plot, pretty_gat, pretty_decod, pretty_slices, pretty_plot
from jr2.stats import gat_stats, parallel_stats

def _my_wilcoxon(X):
    out = wilcoxon(X)
    return out[1]
    
####################Define important variables####################
#Paths
data_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/Loc-loc/IndRes'
indResult_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/Loc-loc/IndRes/Figures'
groupResult_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/Loc-loc/GroupRes'

#List of parameters
ListSubject = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
    'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477', 
    'bo160176', 'at140305', 'df150397', 'rl130571', 'mm140137']

ListCondition = [['Train_loc', 'Test_loc']]


chance = 0

####################Initialize results####################
all_scores = []
all_diagonals = []

####################Load data####################
for c, cond in enumerate(ListCondition):
    for s, subject in enumerate(ListSubject):
        fname = data_path + '/' + subject + '_' + cond[0] + '_' + cond[1] + '-score.npy'
        score = np.load(fname)
        all_scores.append(score)
        
        fname = data_path + '/' + subject + '_' + cond[0] + '_' + cond[1] + '-diagonal.npy'
        diagonal = np.load(fname)
        all_diagonals.append(diagonal)
    
        fname = data_path + '/' + subject + '_' + cond[0] + '_' + cond[1] + '-time.npy'
        time = np.load(fname)
        
        fname = data_path + '/' + subject + '_' + cond[0] + '_' + cond[1] + '-time-loc.npy'
        time_loc = np.load(fname)

all_scores = np.array(all_scores) #shape: subjects*n_cond, training_times, testing_times
all_diagonals = np.array(all_diagonals)  

####################Reshape data####################
all_scores = np.reshape(all_scores, (len(ListCondition), len(ListSubject), 226, 226)) #n_cond, n_subj, training_times, testing_times
all_diagonals = np.reshape(all_diagonals, (len(ListCondition), len(ListSubject), 226))

####################Compute stats for each condition separately####################
p_values_gat = np.zeros((len(ListCondition), all_scores.shape[2], all_scores.shape[3]))
p_values_gat_fdr = np.zeros((len(ListCondition), all_scores.shape[2], all_scores.shape[3]))
p_values_diagonal = np.zeros((len(ListCondition), all_diagonals.shape[2]))
p_values_diagonal_fdr = np.zeros((len(ListCondition), all_diagonals.shape[2]))

for c, cond in enumerate(ListCondition):
    p_values_gat[c, :, :] = parallel_stats(all_scores[c, :, :, :] - chance, function = _my_wilcoxon, correction = False, n_jobs = -1)
    p_values_gat_fdr[c, :, :] = parallel_stats(all_scores[c, :, :, :] - chance, function = _my_wilcoxon, correction = 'FDR', n_jobs = -1)
    p_values_diagonal[c, :] = parallel_stats(all_diagonals[c, :, :] - chance, function = _my_wilcoxon, correction = False, n_jobs = -1)
    p_values_diagonal_fdr[c, :] = parallel_stats(all_diagonals[c, :, :] - chance, function = _my_wilcoxon, correction = 'FDR', n_jobs = -1)
    
    #Get one-sided p-value
    p_values_diagonal[c, :] = p_values_diagonal[c, :]/2.
    p_values_diagonal_fdr[c, :] = p_values_diagonal_fdr[c, :]/2.
    
####################Compute group averages####################
group_scores = np.zeros((len(ListCondition), all_scores.shape[2], all_scores.shape[3]))
sem_group_scores = np.zeros((len(ListCondition), all_scores.shape[2], all_scores.shape[3]))
group_diagonal = np.zeros((len(ListCondition), all_diagonals.shape[2]))
sem_group_diagonal = np.zeros((len(ListCondition), all_diagonals.shape[2]))

for c, cond in enumerate(ListCondition):
    group_scores[c, :, :] = np.mean(all_scores[c, :, :, :], 0)
    sem_group_scores[c, :, :] = stats.sem(all_scores[c, :, :, :], 0)

    group_diagonal[c, :] = np.mean(all_diagonals[c, :, :], 0)
    sem_group_diagonal[c, :] = stats.sem(all_diagonals[c, :, :], 0)

##### Plot GAT with uncorrected p values####################
for c, cond in enumerate(ListCondition):
    pretty_gat(group_scores[c, :, :], times = time_loc, chance = chance, ax = None, sig = None, cmap = 'RdBu_r',
             clim = None, colorbar = True, xlabel = 'Testing Time (s)', 
             ylabel = 'Training Time (s)', sfreq = 250, diagonal = 'dimgray', test_times = None)
    fname = groupResult_path + '/' + cond[0] + '_' + cond[1] + '-uncorrectedGAT_orig.tif'
    plt.savefig(fname, format = 'tif', dpi = 300, bbox_inches = 'tight')
    plt.show()
