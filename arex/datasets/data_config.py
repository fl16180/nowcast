import pandas as pd
import numpy as np
from functools import reduce

TS_VAR = 'Timestamp'


class TSConfig(object):
    """ Register and preprocess time series from arbitrary datasets.
    
    This simplifies the process of combining datasets from different domains.
    The data is unified into a single configuration object which is then
    handled directly by time series models. The main features include:

        1. Register each dataset so that TSConfig can automatically merge
            on timestamp. Arbitrary combinations of datasets/variables can
            be specified for modeling.
        2. Conveniently add autoregressive (lag) terms for any variable from
            any dataset, with extension to the future as far as the data goes.
            Unlike the default pandas lag function which truncates the data at
            the last date, this allows prediction on the future
            (i.e. forecasting).
        3. Simulate forecasting or other information lag at the variable level.
            In other words, rows can be shifted so that the prediction for each
            timestamp uses only what information would have been available at
            the forecast time. 

    Example:
        Suppose we have a target variable in the dataframe 'cdc', and a
        predictor dataset in the dataframe 'external'. First register the
        data:

        >>> dc = TSConfig()
        >>> dc.register_dataset(cdc, 'CDC', 'target')
        >>> dc.register_dataset(external, 'pred', 'predictor')

        Add lag terms of the target variable as autoregressive predictors: 

        >>> dc.add_AR(range(1, 7), dataset='CDC', var_names='%ILI')
        
        Call the stack method to combine the datasets
        >>> dc.stack()

        Combined dataframes (as an (X, y) tuple) can then be accessed using:
        >>> dc.data
    """
    def __init__(self):
        self.datasets = {}
        self.target = None
        self.predictors = []
        self.prepared_data = None

    def register_dataset(self, data, name, type, var_names=None):
        """ Register a dataset with the TSConfig.

        Inputs:
            data (pd.DataFrame): Must contain at least 2 columns, a timestamp
                column (called 'Timestamp'), and a variable column.

            name (str): Name to associate with the dataset

            type (str): 'target' or 'predictor'

            var_names (list): Optional, list of columns to keep
        """
        if type == 'target':
            data = self._register_target(data, name, var_names)
        elif type == 'predictor':
            data = self._register_predictor(data, name, var_names)
        else:
            raise ValueError("type must be 'target' or 'predictor'")
        self.datasets[name] = data

    def _register_target(self, data, name, var_names=None):
        assert TS_VAR in data.columns, 'dataset missing date var'
        data[TS_VAR] = pd.to_datetime(data[TS_VAR])
        data.sort_values(by=TS_VAR, inplace=True)
        data.set_index(TS_VAR, inplace=True)

        if var_names:
            assert isinstance(var_names, str), \
                'only a single target per dataset currently supported'
        else:
            assert data.shape[1] == 1, 'var_names must be specified for >1 col'
            var_names = data.columns[data.columns != TS_VAR][0]
        
        # select column as dataframe
        data = data.loc[:, [var_names]]
        self.target = name
        return data

    def _register_predictor(self, data, name, var_names=None):
        assert TS_VAR in data.columns, 'dataset missing date var'
        data[TS_VAR] = pd.to_datetime(data[TS_VAR])
        data.sort_values(by=TS_VAR, inplace=True)
        data.set_index(TS_VAR, inplace=True)

        if not var_names:
            var_names = data.columns[data.columns != TS_VAR].tolist()
        
        # ensure var_names is a list for dataframe selection
        if isinstance(var_names, str):
            var_names = [var_names]

        data = data.loc[:, var_names]
        self.predictors.append(name)
        return data

    def add_AR(self, terms, dataset, var_names):
        """ Creates an autoregressive (lagged) dataset as additional features.
        These are stored using the name 'AR_dataset'.

        Inputs:
            terms (list): lag terms to consider, e.g. 1 means 1 period ago.

            dataset (str): Dataset containing the variables to be lagged

            var_names (str or list): variables to lag
        """
        # ensure var_names is a list for dataframe selection
        if isinstance(var_names, str):
            var_names = [var_names]

        base_df = self.datasets[dataset].loc[:, var_names]

        # infer time between entries
        tdelta = base_df.index.inferred_freq
        assert tdelta, 'Dataset has gaps or otherwise unable to infer freq'

        # add additional empty rows for extended data
        extend_range = pd.date_range(base_df.index[-1],
                                     periods=max(list(terms)) + 1,
                                     freq=tdelta)[1:]

        extend_df = pd.DataFrame(index=extend_range)
        start_data = pd.concat([base_df, extend_df], sort=True)

        # concatenate all lags
        ar_predictors = pd.concat(
            [start_data.shift(x).add_suffix(f'_AR{x}') for x in terms],
            axis=1)

        name = f'AR_{dataset}'
        self.datasets[name] = ar_predictors
        self.predictors.append(name)

    def stack(self, predictors='all', merge_type='outer', fill_na='ignore'):
        """ Merge datasets together into final modeling dataframe
        
        Inputs:
            predictors ('all' or list): All predictor datasets to use

            merge_type (str): pandas merge option. In general use 'outer'.

            fill_na (str): How to handle missing data.
        """
        if predictors == 'all':
            dfs = [self.datasets[x] for x in self.predictors]
        elif isinstance(predictors, list):
            dfs = [self.datasets[x] for x in predictors]
        self.prepared_data = reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True,
                                  how=merge_type), dfs)
        self.prepared_data.sort_index(inplace=True)
        if fill_na != 'ignore':
            raise NotImplementedError

    @property
    def data(self):
        if self.prepared_data is None:
            raise ValueError('Call stack method before returning data.')
        return self.prepared_data, self.datasets[self.target]