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
        3. Simulate information lag at the variable level.
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
        self.forecast_delay = 0
        self.ar_set = False

    def register_dataset(self, data, name, type, var_names=None, copy=True):
        """ Register a dataset with the TSConfig.

        Inputs:
            data (pd.DataFrame): Must contain at least 2 columns, a timestamp
                column (called 'Timestamp'), and a variable column.

            name (str): Name to associate with the dataset

            type (str): 'target' or 'predictor'

            var_names (list): Optional, list of columns to keep

            copy (bool): Make a copy of data before modifying. If set to false,
                the original dataframe will be altered (only recommend
                when memory is a constraint)
        """
        if copy:
            data = data.copy()
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

    def _extend_date_range(self, df, n_periods):
        """ the standard pandas shift cuts off lags at the last timestamp, but
        we instead want to extend the time index to fully fit the lags. """
        # infer time between entries
        tdelta = df.index.inferred_freq
        assert tdelta, 'Dataset has gaps or otherwise unable to infer freq'

        # add additional empty rows for extended data
        extend_range = pd.date_range(df.index[-1],
                                     periods=n_periods + 1,
                                     freq=tdelta)[1:]

        extend_df = pd.DataFrame(index=extend_range)
        new_df = pd.concat([df, extend_df], sort=True)
        return new_df

    def set_delay(self, periods, datasets='all'):
        """ Specify information delays by dataset.

        This simulates delays in receiving a data source. Here, one can
        specify which datasets are delayed. Caution: Make sure whether AR lags
        based on the target variable should be delayed based on the situation.
        Setting datasets='all' will delay these lags as well, which may not
        be realistic for knowledge delays. This setting is similar to
        forecasting with an important difference: The training set will run up
        to the present, whereas in forecasting the training set is limited to
        the most recent available target value.

        Because of the interaction with the delays and AR terms, set_delay
        methods must be called before add_AR.

        Inputs:
            periods (int): number of time intervals of delay

            datasets (str or list): list of datasets to apply delay or 'all'
                to delay all including AR lags of the target variable.
        """
        periods = int(periods)

        if self.ar_set:
            raise RuntimeError('set_delay must be used before add_AR')
        if datasets == 'all':
            datasets = self.predictors
            self.forecast_delay = periods
        else:
            assert isinstance(datasets, list), 'Pass a list of datasets.'

        for ds in datasets:
            tmp_df = self.datasets[ds].copy()
            extend_df = self._extend_date_range(tmp_df, periods)
            shifted_df = extend_df.shift(periods)

            shifted_df.index.rename(TS_VAR, inplace=True)
            self.datasets[ds] = shifted_df

    def add_AR(self, terms, dataset, var_names='all'):
        """ Creates an autoregressive (lagged) dataset as additional features.
        These are stored using the name 'AR_dataset'.

        Inputs:
            terms (list): lag terms to consider, e.g. 1 means 1 period ago.

            dataset (str): Dataset containing the variables to be lagged

            var_names (str or list): variables to lag. 'all' to select all.
        """
        self.ar_set = True

        # ensure var_names is a list for dataframe selection
        if isinstance(var_names, str):
            if var_names == 'all':
                var_names = self.datasets[dataset].columns
            else:
                raise ValueError("Pass 'all' or a list of variables.")

        # if lag is for target, add forecasting delay to lag
        # other datasets already have delay applied.
        if dataset == self.target:
            shift = max(terms) + self.forecast_delay
            terms = [x + self.forecast_delay for x in terms]
        else:
            shift = max(terms)

        base_df = self.datasets[dataset].loc[:, var_names]
        start_data = self._extend_date_range(base_df, shift)

        # concatenate all lags
        ar_predictors = pd.concat(
            [start_data.shift(x).add_suffix('_lag{0}'.format(x)) for x in terms],
            axis=1)

        ar_predictors.index.rename(TS_VAR, inplace=True)
        name = 'AR_{0}'.format(dataset)
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
            raise RuntimeError('Call stack method before returning data.')
        return self.prepared_data, self.datasets[self.target]