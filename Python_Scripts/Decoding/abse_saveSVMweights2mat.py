import pickle
import numpy as np
from os import path
from scipy.io import savemat

subjects = ['am150105', 'bb130599', 'cd130323', 'jf140150', 'ql100269', 
	'sb150103', 'sl130503','ws140212', 'jl150086', 'fm100109', 
	'hb140194', 'sd150012', 'mk140057', 'xm140202', 'lr110094']

data_path = '/neurospin/meg/meg_tmp/ABSE_Marti_2014/mne/decod_tmp/'
save_path = '/neurospin/meg/meg_tmp/ABSE_Marti_2014/mat/weights/'

for i_sub in range(len(subjects)):
	filename = subjects[i_sub] + '_decod_train_fit_v4.p'
	gat = pickle.load(open(path.join(data_path, filename), 'rb'))

	n_class = 4
	n_time = 61
	n_chan = 306
	n_fold = 5
	coef = np.zeros((n_chan,n_time,n_class,n_fold))

	for i_time in range(n_time):
		for i_class in range(n_class):
			for i_fold in range(n_fold):
				coef[:,i_time,i_class,i_fold] = gat.estimators_[i_time][i_fold].named_steps['onevsrestclassifier'].coef_[i_class]
	
	savename = save_path + filename[0:-2] + '.mat'
	savemat(savename, dict(coef=coef))
