import numpy as np
import pandas as pd
from forecastlib import TS_VAR

CDC_REGIONS = './Regions.csv'
MMWR_LOOKUP = './MMWR_lookup.csv'


class CDCLoader(object):
    """ Loader for CDC data.

    Example usage:
        >>> cdcl = CDCLoader("./ILI_national_dated.csv")
        >>> cdc = cdcl.load_national()
    """
    DATE_VAR = 'Date'
    ILI_WEIGHTED = '% WEIGHTED ILI'
    ILI_UNWEIGHTED = '%UNWEIGHTED ILI'
    REGION_ID = 'REGION'
    STATE_ID = 'State'
    ILI_RENAME = "%ILI"

    def __init__(self, filename, ili_version='weighted'):
        self.data = pd.read_csv(filename)
        if ili_version == 'weighted':
            self.data[self.ILI_RENAME] = self.data[self.ILI_WEIGHTED]
        elif ili_version == 'unweighted':
            self.data[self.ILI_RENAME] = self.data[self.ILI_UNWEIGHTED]

    def load_national(self):
        data = self.data
        data[TS_VAR] = pd.to_datetime(data[self.DATE_VAR])
        return data[[TS_VAR, self.ILI_RENAME]].copy()

    def load_regional(self, region):
        data = self.data
        data[TS_VAR] = pd.to_datetime(data[self.DATE_VAR])
        data[self.REGION_ID] = data[self.REGION_ID].map(lambda x: x[7:])
        data = data[data[self.REGION_ID] == region]
        return data[[TS_VAR, self.ILI_RENAME]].copy()

    def load_state(self, state):
        data = self.data
        data[TS_VAR] = pd.to_datetime(data[self.DATE_VAR])
        data = data[data[self.STATE_ID] == state]
        return data[[TS_VAR, self.ILI_RENAME]].copy()


class AthenaLoader(object):
    """ Loader for CDC data.

    Example usage:
        >>> athl = AthenaLoader("./ATHdata.csv")
        >>> ath = athl.load_national()
    """
    DATE_VAR = 'Week Start Date'
    ATHENA_VARS = ['Flu Visit Count', 'ILI Visit Count',
                   'Unspecified Viral or ILI Visit Count']
    COUNTS = 'Visit Count'

    def __init__(self, filename, smoothing=None):
        self.data = pd.read_csv(filename)
        self.data[TS_VAR] = pd.to_datetime(self.data[self.DATE_VAR])
        if self.DATE_VAR != TS_VAR:
            self.data.drop(self.DATE_VAR, axis=1, inplace=True)
        self.region_data = None
        self.smoothing = smoothing

    def load_national(self):
        data = self.data[self.data['State'] == 'ALL STATES'].copy()
        return self._process_data(data)

    def load_regional(self, region):
        # merge with CDC region lookup
        if not self.region_data:
            regions = pd.read_csv(CDC_REGIONS)
            region_data = pd.merge(self.data, regions, how='left', on='State')
            region_data = region_data.groupby(['Region', 'Year', 'MMWR Week'],
                                              as_index=False).sum()
            self.region_data = region_data

        data = self.region_data[self.region_data['Region'] == region].copy()
        return self._process_data(data)

    def load_state(self, state):
        data = self.data[self.data['State'] == state].copy()
        return self._process_data(data)

    def _process_data(self, data):
        if self.smoothing == 'moving_avg':
            raise NotImplementedError
        else:
            data[self.ATHENA_VARS] = \
                data[self.ATHENA_VARS].div(data[self.COUNTS], axis=0)
        return data[[TS_VAR] + self.ATHENA_VARS].copy()


def gt_loader(filename):
    """ Loader for Google Trends data """
    data = pd.read_csv(filename)
    data[TS_VAR] = pd.to_datetime(data['date'])
    data.drop('date', axis=1, inplace=True)
    return data
