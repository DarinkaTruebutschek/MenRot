###Purpose: This function converts EOG data from a fieldtrip structure to mne epochs. 
###Project: General
###Author: Darinka Truebutschek
###Date: 11 October 2018

#Load libraries
import mne 
import numpy as np
import scipy.io as sio

from mne.io.meas_info import create_info
from mne.epochs import EpochsArray

def fldtrp2mne_EOG(filename, var):
    "This function converts data from a fieldtrip structure to mne epochs"
    
    #Load Matlab/fieldtrip data
    mat = sio.loadmat(filename, squeeze_me = True, struct_as_record = False)
    ft_data = mat[var]
    
    #Identify basic parameters
    n_trial = len(ft_data.ECGEOG)
    n_chans, n_time = ft_data.ECGEOG[0].shape #ATTENTION: This presumes equal epoch lengths across trials
    data = np.zeros((n_trial, n_chans-1, n_time))
    
    for triali in range(n_trial):
        data[triali, :, :] = ft_data.ECGEOG[triali][0:1, :] #select only the EOG data
    
    #Add all necessary info
    sfreq = float(ft_data.fsample) #sampling frequency
    #lpFilter = float(ft_data.cfg.lpfreq)
    time = ft_data.time[0];
    coi = range(2) #channels of interest
    data = data[:, coi, :]
    #chan_names = [l.encode('ascii') for l in ft_data.label[coi]] #this always appends the letter b before the actual filename?
    #chan_names = ft_data.label.tolist()
    chan_names = list(['EOG1', 'EOG2'])
    chan_types = ft_data.label[coi]
    chan_types[0] = 'eog'
    chan_types[1] = 'eog'
    #chan_types[2] = 'ecg'
    
    #Create info and epochs
    info = create_info(chan_names, sfreq, chan_types)
    events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                   np.zeros(n_trial), np.zeros(n_trial)]
    
    epochs = EpochsArray(data, info, events = np.array(events, int), 
                         tmin = np.min(time), verbose = False)
                         
    return epochs
