''' argo_test.py
Runs the basic ARGO algorithm to make time series predictions using exogenous data input.

Reads in two csv files: Google Trends data and official dengue case count data.
Script assumes that the csvs are in the following format:
    Google Trends:
    column 1 is time (week/month/etc), columns 2-end are each a Google query term.
    case count data:
    column 1 is time, column 2 is case count numbers.

    The two csv files should also be aligned to the same time window
    (i.e. column 1 of both should be the same).
    If not, either modify the csvs or modify the indexing in the code after reading in the files
    so that they are aligned before passing into the ARGO function.

'''

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import KFold
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import sys

# insert working directory here (where data files are located)
HOME_DIR = ' insert here '


# scoring functions
def rmse(predictions, targets):
        # root mean square error
    return sqrt(((predictions - targets) ** 2).mean())


def corr(predictions, targets):
        # pearson r
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]


def mae(predictions, targets):
    return np.absolute(predictions - targets).mean()


def report_stats(predictions, targets):
    print "rmse: ", rmse(predictions, targets)
    print "corr: ", corr(predictions, targets)
    print "mae: ", mae(predictions, targets)


def ARGO(X, y, ahead):
    ''' Defines the basic ARGO algorithm. ARGO is a multivariate linear regression model combining multiple information sources,
    with L1 regularization.
        Params:
        X: numpy matrix of independent variables
        y: dependent variable
        'ahead': specifier indicating forecast horizon (0 = nowcast, 1 = 1 wk forecast). 

        length(X) should be length(y) or length(y)+1.
        ARGO returns a prediction array of the same length as X minus a 104 week training period.

        Note: X is n x p matrix of n observations and p features, y is n x 0 or n x 1 matrix.
    '''
    np.random.seed(1500)    # set seed

    # initialize output array
    predictions = np.zeros(len(X) - 104 - ahead)

    # historical predictions training on previous 104 weeks
    for j in range(104 + ahead, len(X)):
        start = max(0, j - (104 + ahead))

        # show progress on console
        sys.stdout.write(str(start))
        sys.stdout.write('\r')
        sys.stdout.flush()

        # standardizes the variables
        scaler = StandardScaler()
        X_n = scaler.fit_transform(X[start:j + 1])

        # create lasso regression training sets and value to test
        X_train = X_n[:-(1 + ahead)]
        y_train = y[start + ahead:j]
        X_test = X_n[-1, None]

        # create 10-fold cross validator object
        folds = KFold(104, n_folds=10, shuffle=True)

        # fit and predict
        Lasso = LassoCV(cv=folds, n_alphas=200, max_iter=20000, tol=.0005, normalize=False)
        predictions[start] = Lasso.fit(X_train, y_train).predict(X_test)

    return predictions


def main():
    ### Load and process data ###

    cases = pd.read_csv(HOME_DIR + 'case_counts.csv', parse_dates=[0])
    google = pd.read_csv(HOME_DIR + 'GT.csv', parse_dates=[0])

    # extract relevant variables (column 2 of case count array, columns 2-end of google data array).
    # before continuing make sure that the two csvs are aligned. else align them in the following code.
    align_truth = 0
    align_gt = 0

    truth = cases.values[align_truth:, 1]
    truth_for_scoring = truth[104:]
    gt = google.values[align_gt:, 1:].astype(float)


    ######## Predictions #########

    # nowcast predictions
    predictions = ARGO(gt, truth, 0)

    # print scoring metrics
    report_stats(predictions, truth_for_scoring)

    # save the predictions
    time = pd.Series(cases.values[align_truth + 104:, 0].flatten(), name='time')
    pred = pd.Series(predictions, name='prediction')
    df = pd.concat([time, pred], axis=1)
    df.to_csv(HOME_DIR + 'output.csv', index=False)


if __name__ == '__main__':
    main()
