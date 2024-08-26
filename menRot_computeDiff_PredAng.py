#Purpose: This script computes the difference in the  distributions of predicted angles 
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 2 April 2018

import numpy as np

from menRot_base_plot import pretty_plot, plot_sem, plot_widths, pretty_colorbar
from menRot_plotGat import _set_ticks

###Define important variables###
ListAnalysis = ['Loc_TrainAll_TestAll']
condition1 = 'NoRotSeen' #seen, unseen, etc ...
condition2 = 'RotSeen'

path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'
res_path = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Decoding/newPipeline/Grouped/FiveFolds/IntermediateLocations/' + ListAnalysis[0] + '/GroupRes/Stats/'

n_bins = np.linspace(-np.pi, np.pi, 25)
times = np.linspace(-0.2, 3.496, 463)

#Load data
data1 = np.load(path + condition1 + '_' + str(len(n_bins)) + '-histogram.npy') #subjects x n_bins x time
data1 = np.array(data1)

data2 = np.load(path + condition2 + '_' + str(len(n_bins)) + '-histogram.npy') #subjects x n_bins x time
data2 = np.array(data2)

data = data1 - data2

#Save data
np.save(res_path + condition1 + '-' + condition2 + '_' + str(len(n_bins)) + '-histogram.npy', data)
