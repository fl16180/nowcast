import numpy as np
import pandas as pd
from tqdm import tqdm
from arex.datasets import TSConfig


class Arex(object):
    """ AREX (AutoRegression with EXogeneity) is an iterative time series
    predictor that abstracts away the logic of retraining a model sequentially
    on time series data. AREX does not impose any modeling constraints --
    instead it is a procedure that can handle any model that is compatible
    with scikit-learn's fit/predict API.

    Usually, one retrains a time series model at each time step in order to use
    the most recent information. The training set at each step can be either
    rolling (fixed size that discards old data), or expanding (use all data).
    Often, a time series is predicted using a combination of lags of the
    time series (AR), concurrent exogenous variables (EX), and lags of the
    exogenous variables. AREX takes care of these details for you.

    On the other hand, the actual model that is applied at each time step is
    highly important to researchers -- it can involve preprocessing and feature
    engineering to using various ML algorithms and hyperparameter tuning. Thus
    this part is flexible and only limited by your creativity. All AREX needs
    is a model class with fit() and predict() methods, identical to sklearn.
    In fact, any sklearn model can be passed directly into AREX to get an
    out-of-the-box time series modeler.

    Example:
        Suppose we continue the example with TSConfig.
        >>> dc = TSConfig()
        >>> dc.register_dataset(cdc, 'CDC', 'target')
        >>> dc.register_dataset(external, 'pred', 'predictor')
        >>> dc.add_AR(range(1, 7), dataset='CDC', var_names='%ILI')
        >>> dc.stack()

        We will use a default sklearn random forest as the model:
        >>> mod = RandomForestRegressor()

        The above time series is at a weekly frequency. For nowcasting
        (predicting target at week t using exogenous data from week t) with
        a year-long rolling training window, do:
        >>> arex = Arex(model=mod, data_config=dc)
        >>> pred = arex.nowcast(pred_start='2019-02-19', pred_end='2019-08-20',
                                training='roll', window=52)

        Suppose we want to predict
    """
    def __init__(self, model, X=None, y=None, data_config=None, verbose=1):
        self.model = model
        self.X = X
        self.y = y
        self.config = data_config
        self.pred_start = None
        self.pred_end = None
        self.training = None
        self.window = None
        self.pred_name = None
        self.verbose = verbose
        self.log = []

        self._validate_init()

        if self.config:
            self.X, self.y = self.config.data

        if self.verbose > 0:
            print('---------------- ArEx model initialized ----------------')

    def forecast(self, t_plus, pred_start, pred_end,
                training, window, pred_name='Predicted', t_known=False):
        self.t_plus = t_plus
        self.pred_start = pd.to_datetime(pred_start)
        self.pred_end = pd.to_datetime(pred_end)
        self.training = training
        self.window = window
        self.pred_name = pred_name
        self.t_known = t_known
        self.log = []

        self._validate_predict()

        if self.verbose > 0:
            print(f'Predicting from {self.pred_start} to {self.pred_end}:')

        # range of timestamps to predict
        pred_range = self.X.loc[self.pred_start:self.pred_end].index

        predictions = []
        for timestamp in tqdm(pred_range):
            X_train_start, X_train_end, y_train_start, y_train_end = \
                self._get_start_end_index(timestamp)

            X_train = self.X.iloc[X_train_start:X_train_end]
            y_train = self.y.iloc[y_train_start:y_train_end, 0]

            X_pred_index = self._get_pred_index(timestamp)
            X_pred = self.X.iloc[X_pred_index:X_pred_index + 1]

            if self.verbose == 2:
                print('Predicting time: ', timestamp)
                print('X_train: ', X_train.index[0], X_train.index[-1])
                print('y_train: ', y_train.index[0], y_train.index[-1])
                print('X_pred: ', X_pred.index[0])
                print('sizes: ', X_train.shape, y_train.shape)

            debug_log = {}
            debug_log['time'] = timestamp
            debug_log['X_train'] = (X_train.index[0], X_train.index[-1])
            debug_log['y_train'] = (y_train.index[0], y_train.index[-1])
            debug_log['X_pred'] = X_pred.index[0]
            debug_log['sizes'] = (X_train.shape, y_train.shape)
            self.log.append(debug_log)

            self.model.fit(X_train, y_train)
            predictions.append(self.model.predict(X_pred)[0])

        pred_df = pd.DataFrame(data={self.pred_name: predictions},
                               index=pred_range)
        return pred_df

    def nowcast(self, pred_start, pred_end,
                training, window, pred_name='Predicted'):
        return self.forecast(t_plus=0, pred_start=pred_start,
                             pred_end=pred_end, training=training,
                             window=window, pred_name=pred_name,
                             t_known=False)

    def get_params(self):
        """ Returns user-set parameters of ArEx object """
        params = {}
        params['pred_start'] = self.pred_start
        params['pred_end'] = self.pred_end
        params['training'] = self.training
        params['window'] = self.window
        params['pred_name'] = self.pred_name
        return params

    def get_log(self):
        return self.log

    def _validate_init(self):
        assert hasattr(self.model, 'fit') and hasattr(self.model, 'predict'), \
            'model must have fit and predict methods.'

        if self.X and self.y:
            assert isinstance(self.X, pd.DataFrame), \
                'X and y must be dataframes'
            assert isinstance(self.y, pd.DataFrame), \
                'X and y must be dataframes'

        if self.config:
            assert isinstance(self.config, TSConfig), \
                'data_config must be a TSConfig object'

        assert self.config or (self.X and self.y), \
            'Either pass X and y dataframes or a TSConfig object'
        if self.config and (self.X and self.y):
            print('Both X and y and TSConfig were passed, will default to '
                  'TSConfig')

    def _validate_predict(self):
        assert self.training in ('expand', 'roll')
        if self.training == 'roll':
            assert self.window > 0
            self.window = int(self.window)

        assert self.pred_start in self.X.index, 'pred_start not in X dataframe'
        assert self.pred_end in self.X.index, 'pred_end not in X dataframe'

    def _get_start_idx(self, pred_timestamp):
        # forecasting shifts train set back
        backshift = self.t_plus - int(self.t_known)

        if pred_timestamp in self.y.index:
            # if target is available at train end
            X_train_end = np.where(self.X.index == pred_timestamp)[0][0]
            y_train_end = np.where(self.y.index == pred_timestamp)[0][0]
            X_train_end = X_train_end - backshift
            y_train_end = y_train_end + int(self.t_known)
        else:
            # find most recent available training timestamp
            last_train_ts = min(self.X.index.max(), self.y.index.max())
            X_train_end = np.where(self.X.index == last_train_ts)[0][0] + 1
            y_train_end = np.where(self.y.index == last_train_ts)[0][0] + 1
            X_train_end = X_train_end - self.t_plus
        return X_train_end, y_train_end

    def _get_start_end_index(self, pred_timestamp):

        # find train end parameters
        X_train_end, y_train_end = self._get_start_idx(pred_timestamp)

        # find train start parameters
        if self.training == 'expand':
            X_pred_start, y_pred_start = self._get_start_idx(self.pred_start)
            X_train_start = X_pred_start - self.window
            y_train_start = y_pred_start - self.window

        else:
            X_train_start = X_train_end - self.window
            y_train_start = y_train_end - self.window

        if min(X_train_start, y_train_start) < 0:
            raise ValueError(
                f'Warning: Train set has {min(X_train_end, y_train_end)} out '
                f'of {self.window} rows needed')

        return X_train_start, X_train_end, y_train_start, y_train_end

    def _get_pred_index(self, pred_timestamp):
        return np.where(self.X.index == pred_timestamp)[0][0]
