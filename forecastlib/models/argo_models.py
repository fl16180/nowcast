import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.special import logit, expit


class Argo(object):
    """ The base ARGO model without extra features, with Lasso regression """
    def __init__(self):
        self.model = LassoCV(eps=0.001, n_alphas=200, cv=10,
                             max_iter=1e4, n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_train = self.scaler.fit_transform(X)
        self.model.fit(X_train, y)

    def predict(self, X):
        X_pred = self.scaler.transform(X)
        return self.model.predict(X_pred)


class Argo2015(object):
    """ The ARGO model based on Yang et al. 2015

        Currently not configured to transform Google Trends data.
        It takes the logit of all inputs.
    """
    def __init__(self):
        self.model = LassoCV(eps=0.001, n_alphas=200, cv=10,
                             max_iter=1e4, n_jobs=-1)
        self.scaler = StandardScaler()

        self.logit = lambda x: logit(x)
        self.expit = lambda x: expit(x)
        self.log = lambda x: np.log(x)

    def fit(self, X, y):
        # ar_cols = [x for x in X.columns if 'AR_' in x]
        # X_ar = X[ar_cols].copy()
        # X[ar_cols] = self.logit(X_ar / 100)

        Xl = logit(X / 100)
        yl = logit(y / 100)
        X_train = self.scaler.fit_transform(Xl)
        self.model.fit(X_train, yl)

    def predict(self, X):
        Xl = self.logit(X / 100)
        X_pred = self.scaler.transform(Xl)
        return self.expit(self.model.predict(X_pred)) * 100


class ArgoSVM(object):
    """ ARGO with SVM regressor, based on Santillana et al. 2016 """
    def __init__(self):
        import warnings
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        warnings.filterwarnings(action='ignore', category=FutureWarning)

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
