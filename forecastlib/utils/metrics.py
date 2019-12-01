""" Simple metrics included for user convenience. For more metrics
import from scikit-learn or define your own. """

import numpy as np
from math import sqrt
from scipy.stats.stats import pearsonr


def RMSE(targets, predictions):
    """ root mean square error """
    return sqrt(((targets - predictions) ** 2).mean())

def Corr(targets, predictions):
    """ Pearson correlation coefficient """
    corr_c = pearsonr(targets, predictions)
    return corr_c[0]

def MAE(targets, predictions):
    """ mean absolute error """
    return np.absolute(targets - predictions).mean()


def MAPE(targets, predictions):
    """ mean absolute percent error """
    return np.mean(np.abs((targets - predictions) / targets))