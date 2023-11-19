import numpy as np
import pandas as pd
from tqdm import tqdm
from nowcast import TSConfig


class Arex(object):
    """ ``AREX`` (AutoRegression with EXogeneity) is an iterative time series
    predictor that abstracts away the logic of retraining a model sequentially
    on time series data. ``AREX`` does not impose any modeling constraints --
    instead it is a procedure that can handle any model that is compatible
    with scikit-learn's fit/predict API.

    Usually, one retrains a time series model at each time step in order to use
    the most recent information. The training set at each step can be either
    rolling (fixed size that discards old data), or expanding (use all data).
    Often, a time series is predicted using a combination of lags of the
    time series (AR), concurrent exogenous variables (EX), and lags of the
    exogenous variables. ``AREX`` takes care of these details for you.

    On the other hand, the actual model that is applied at each time step is
    highly important to researchers -- it can involve preprocessing and feature
    engineering to using various ML algorithms and hyperparameter tuning. Thus
    this part is flexible and only limited by your creativity. All AREX needs
    is a model class with ``fit()`` and ``predict()`` methods, identical to
    sklearn.

    In fact, any sklearn model can be passed directly into ``AREX`` to get an
    out-of-the-box time series modeler.

    Example:
        Suppose we continue the example with ``TSConfig``::

            $ dc = TSConfig()
            $ dc.register_target(cdc, 'Date', 'CDC')
            $ dc.register_dataset(external, 'pred', 'Date', 'predictor')
            $ dc.add_AR(range(1, 7), dataset='CDC', var_names='%ILI')
            $ dc.stack()

        We will use a default sklearn random forest as the model::
            $ mod = RandomForestRegressor()

        The above time series is at a weekly frequency. For nowcasting
        (predicting target at week t using exogenous data from week t) with
        a year-long rolling training window, do::

            $ arex = Arex(model=mod, data_config=dc)
            $ pred = arex.nowcast(pred_start='2019-02-19',
                                  pred_end='2019-08-20',
                                  training='roll', window=52)

        Suppose we want to predict a week ahead. We would run::

            $ pred2 = arex.forecast(t_plus=1, pred_start='2019-02-19',
                                    pred_end='2019-08-20',
                                    training='roll', window=52)

        Note that the timestamps for pred_start and pred_end refer to the time
        of making the prediction, not the time that is predicted.
        
        The returned prediction dataframe will contain a column called
        "Timestamp", which is the time of making the prediction, a column
        called "Event Time", which is the time of the predicted event, and
        the predicted column entitled "Predicted".

    Args:
        model (class): Any model with ``fit`` and ``predict`` methods following
            sklearn API. The methods must accept pandas dataframes.

        X (dataframe): Predictor dataframe. Either pass dataframes for both
            ``X`` and ``y``, or otherwise pass a TSConfig object to
            data_config.

        y (dataframe): Target dataframe with a single column with the target
            measurements, indexed by timestamp

        data_config (TSConfig): TSConfig object containing preprocessed
            ``X`` and ``y`` data using TSConfig.
            This will overwrite ``X`` and ``y`` if passed.

        verbose (int): Control verbosity of output
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
            self.period = self.config.period
        else:
            self.period = TSConfig._compute_period(self.y)

        if self.verbose > 0:
            print('---------------- AREX model initialized ----------------')

    def forecast(self, t_plus, pred_start, pred_end,
                training, window, pred_name='Predicted', t_known=False):
        """ Perform rolling forecast on data.

        Args:
            t_plus (int): Number of periods ahead to forecast. 0 corresponds to
                nowcasting (predict ``y_t`` from ``X_t``).

            pred_start (str or datetime): Time of first prediction

            pred_end (str or datetime): Time of last prediction

            training (str): Training window behavior, either 'roll' or 'expand'

            window (int): Training window size. If training is 'expand', then
                window determines the size for the first prediction, and
                subsequent windows increase in size.

            pred_name (str): Name for prediction output column

            t_known (bool): Whether ``y_t`` would be known at time of
                forecasting. This adjusts the training window to use the most
                recent information. Since ``y_t`` is the nowcast target,
                ``t_known`` must be set to False when ``t_plus=0``.
        """
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
            print('Predicting from {0} to {1}:'.format(self.pred_start,
                                                       self.pred_end))

        # range of timestamps to predict
        pred_range = self.X.loc[self.pred_start:self.pred_end].index
        event_range = pred_range + self.period * self.t_plus

        predictions = []
        for i in tqdm(range(len(pred_range))):
            timestamp = pred_range[i]
            time_event = event_range[i]

            X_train_start, X_train_end, y_train_start, y_train_end = \
                self._get_start_end_index(timestamp)

            X_train = self.X.iloc[X_train_start:X_train_end]
            y_train = self.y.iloc[y_train_start:y_train_end, 0]

            X_pred_index = self._get_pred_index(timestamp)
            X_pred = self.X.iloc[X_pred_index:X_pred_index + 1]

            if self.verbose == 2:
                print('Time prediction made: ', timestamp)
                print('Time of event: ', time_event)
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

        pred_df = pd.DataFrame(data={'Event Time': event_range,
                                     self.pred_name: predictions},
                               index=pred_range).reset_index()
        return pred_df

    def nowcast(self, pred_start, pred_end,
                training, window, pred_name='Predicted'):
        """ Perform rolling nowcast on data. This is just a convenience wrapper
        for the forecast method.
        """
        return self.forecast(t_plus=0, pred_start=pred_start,
                             pred_end=pred_end, training=training,
                             window=window, pred_name=pred_name,
                             t_known=False)

    def get_params(self):
        """ Returns user-set parameters of ``Arex`` object """
        params = {}
        params['pred_start'] = self.pred_start
        params['pred_end'] = self.pred_end
        params['training'] = self.training
        params['window'] = self.window
        params['pred_name'] = self.pred_name
        return params

    def get_log(self):
        """ Return a dataframe of metadata for each prediction. For example,
        the ``X_train`` and ``y_train`` date ranges, prediction time, and
        training set sizes. Use this for debugging or to confirm behavior.
        """
        return pd.DataFrame(self.log)

    def _validate_init(self):
        assert hasattr(self.model, 'fit') and hasattr(self.model, 'predict'), \
            'model must have fit and predict methods.'

        if self.X is not None and self.y is not None:
            assert isinstance(self.X, pd.DataFrame), \
                'X and y must be dataframes'
            assert isinstance(self.y, pd.DataFrame), \
                'X and y must be dataframes'

        if self.config:
            assert isinstance(self.config, TSConfig), \
                'data_config must be a TSConfig object'

        assert self.config or (self.X is not None and self.y is not None), \
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
        # forecasting shifts train set back, determine the correct amount
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
                'Train set has {0} out of {1} rows needed'.format(
                    min(X_train_end, y_train_end), self.window))

        return X_train_start, X_train_end, y_train_start, y_train_end

    def _get_pred_index(self, pred_timestamp):
        return np.where(self.X.index == pred_timestamp)[0][0]
