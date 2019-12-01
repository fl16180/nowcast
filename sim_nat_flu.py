from forecastlib.datasets import CDCLoader, AthenaLoader
from forecastlib import TSConfig, Arex
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

cdcl = CDCLoader("./ILI_national_dated.csv", ili_version='weighted')
cdc = cdcl.load_national()

athl = AthenaLoader("./ATHdata.csv")
ath = athl.load_national()

dc = TSConfig()
dc.register_dataset(cdc, 'CDC', 'target')
dc.register_dataset(ath, 'athena', 'predictor')
dc.add_AR(range(1, 53), dataset='CDC', var_names=['%ILI'])
dc.stack()


class Argo(object):
    """ The base ARGO model without extra features """
    def __init__(self):
        self.model = LassoCV(eps=0.001, n_alphas=150, cv=10,
                             max_iter=1e4, n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_train = self.scaler.fit_transform(X)
        self.model.fit(X_train, y)

    def predict(self, X):
        X_pred = self.scaler.transform(X)
        return self.model.predict(X_pred)


class Argo2015(object):
    """ The base ARGO model without extra features """
    def __init__(self):
        from scipy.special import logit, expit

        self.model = LassoCV(eps=0.001, n_alphas=200, cv=10,
                             max_iter=1e4, n_jobs=-1)
        self.scaler = StandardScaler()

        self.logit = lambda x: logit(x)
        self.expit = lambda x: expit(x)

    def fit(self, X, y):
        Xl = self.logit(X / 100)
        yl = self.logit(y / 100)
        X_train = self.scaler.fit_transform(Xl)
        self.model.fit(X_train, yl)

    def predict(self, X):
        Xl = self.logit(X / 100)
        X_pred = self.scaler.transform(Xl)
        return self.expit(self.model.predict(X_pred)) * 100

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from scipy.special import logit, expit

class ArgoSVM(object):
    def __init__(self):
        C_range = 10.0 ** np.arange(-3, 3)
        gamma_range = 10.0 ** np.arange(-3, 3)
        epsilon_range = 10.0 ** np.arange(-1, 1)
        self.params = dict(kernel=['rbf'],
                           epsilon=epsilon_range,
                           gamma=gamma_range,
                           C=C_range)
        self.model = GridSearchCV(SVR(), self.params, cv=10, n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xl = logit(X / 100)
        yl = logit(y / 100)
        X_train = self.scaler.fit_transform(Xl)
        self.model.fit(X_train, yl)

    def predict(self, X):
        Xl = logit(X / 100)
        X_pred = self.scaler.transform(Xl)
        return expit(self.model.predict(X_pred)) * 100

class MockModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return [1]

# base_mod = MockModel()
base_mod = Argo2015()
# base_mod = ArgoSVM()
arex = Arex(model=base_mod, data_config=dc)
# p1 = arex.nowcast(pred_start='2019-10-20', pred_end='2019-11-17',
#                   training='roll', window=104, pred_name='ARGO 1wk')

# p1 = arex.forecast(t_plus=1, pred_start='2019-10-13', pred_end='2019-11-17',
#                   training='roll', window=104, pred_name='ARGO 2wk')

p1 = arex.forecast(t_plus=2, pred_start='2019-10-06', pred_end='2019-11-17',
                  training='roll', window=104, pred_name='ARGO 3wk')


p1 = arex.forecast(t_plus=3, pred_start='2019-09-29', pred_end='2019-11-17',
                  training='roll', window=104, pred_name='ARGO 4wk')

print(p1)
ll = arex.get_log()
for i in ll:
    print(i)