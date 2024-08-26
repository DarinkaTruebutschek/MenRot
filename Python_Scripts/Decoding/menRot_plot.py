###Load necessary libraries###

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import wilcoxon

#Add personal functions to python path
sys.path.append('/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/Decoding/')

from jr.plot import base, gat_plot, pretty_gat, pretty_decod, pretty_slices
from jr.stats import gat_stats, parallel_stats
