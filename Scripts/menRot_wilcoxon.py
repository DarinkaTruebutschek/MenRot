###Purpose: Compute Wilcoxon signed-rank test
###Project: MenRot
###Author: Darinka Trubutschek
###Date: 18 December 2017

from scipy import stats
from scipy.stats import wilcoxon

def _my_wilcoxon(X):
	out = wilcoxon(X)
	return out[1]