import numpy as np
import pandas as pd
from tqdm import tqdm


class Arex(object):
    def __init__(self, model, X=None, y=None, data_config=None, verbose=True):
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

        self._validate_init()

        if self.config:
            self.X, self.y = self.config.data

        if self.verbose:
            print('---------------- ArEx model initialized ----------------')

    def predict(self, pred_start, pred_end, 
                training, window, pred_name='Predicted'):
        self.pred_start = pd.to_datetime(pred_start)
        self.pred_end = pd.to_datetime(pred_end)
        self.training = training
        self.window = window
        self.pred_name = pred_name

        self._validate_predict()

        if self.verbose:
            print(f'Predicting from {self.pred_start} to {self.pred_end}:')

        # range of timestamps to predict
        pred_range = self.X[(self.X.index >= self.pred_start) &
                            (self.X.index <= self.pred_end)].index
        
        predictions = []
        for timestamp in tqdm(pred_range):
            X_train_start, X_train_end, y_train_start, y_train_end = \
                self._get_start_end_index(timestamp)

            X_train = self.X.iloc[X_train_start:X_train_end]
            y_train = self.y.iloc[y_train_start:y_train_end]

            X_pred_index = self._get_pred_index(timestamp)
            X_pred = self.X.iloc[X_pred_index:X_pred_index + 1]

            self.model.fit(X_train, y_train)
            predictions.append(self.model.predict(X_pred))

        pred_df = pd.DataFrame(data={self.pred_name: predictions},
                               index=pred_range)
        return pred_df

    def get_params(self):
        """ Returns user-set parameters of ArEx object """
        params = {}
        params['pred_start'] = self.pred_start
        params['pred_end'] = self.pred_end
        params['training'] = self.training
        params['window'] = self.window
        params['pred_name'] = self.pred_name
        return params

    def _validate_init(self):
        assert hasattr(self.model, 'fit') and hasattr(self.model, 'predict'), \
            'model must have fit and predict methods.'
        
        if self.config:
            assert isinstance(self.config, TSConfig), \
                'data_config must be a TSConfig object'

        assert self.config or (X and y), \
            'Either pass X and y dataframes or a TSConfig object'
        if self.config and (X and y):
            print('Both X and y and TSConfig were passed, will default to '
                  'TSConfig')

    def _validate_predict(self):
        assert self.training in ('expand', 'roll')
        if self.training == 'roll':
            assert self.window > 0
            self.window = int(self.window)

        assert self.pred_start in self.X.index
        assert self.pred_end in self.X.index

    def _get_start_end_index(self, pred_timestamp):
        
        if pred_timestamp in self.y.index:
            # if target is available at train end
            X_train_end = np.where(self.X.index == pred_timestamp)[0]
            y_train_end = np.where(self.y.index == pred_timestamp)[0]
        else:
            # find most recent available training timestamp
            last_train_timestamp = max(self.X.index.max(), self.y.index.max())
            X_train_end = np.where(self.X.index == last_train_timestamp)[0] + 1
            y_train_end = np.where(self.y.index == last_train_timestamp)[0] + 1

        # find train start based on train parameters
        if self.training == 'expand':
            X_train_start = 0
        else:
            X_train_start = max(0, X_train_end - self.window)
            if X_train_end - self.window < 0:
                print(f'Warning: Train set has {X_train_end - self.window} '
                      f'out of {self.window} rows needed')
        
        train_start_timestamp = self.X.index[X_train_start]
        if train_start_timestamp not in self.y.index:
            raise ValueError(f'No target entry for {train_start_timestamp}')
        y_train_start = np.where(self.y.index == train_start_timestamp)[0]
        return X_train_start, X_train_end, y_train_start, y_train_end

    def _get_pred_index(self, pred_timestamp):
        return np.where(self.X.index == pred_timestamp)[0]

