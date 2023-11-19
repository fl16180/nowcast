import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from nowcast import TSConfig, TS_VAR


DATES = np.array(['2019-01-01', '2019-01-08', '2019-01-15',
                  '2019-01-22', '2019-01-29', '2019-02-05',
                  '2019-02-12', '2019-02-19'])
X = np.array([2, 3, 4, 4, 5, 4, 3, 1])
Y = np.array([3, 5, 6, 6, 3, 2, 4, 1])

TRAIN_DF = pd.DataFrame({TS_VAR: DATES, 'x': X})
TARGET_DF = pd.DataFrame({TS_VAR: DATES, 'y': Y})

CFG_X_DF = pd.DataFrame({'x': X}, index=pd.to_datetime(DATES))
CFG_X_DF.index.rename(TS_VAR, inplace=True)
CFG_Y_DF = pd.DataFrame({'y': Y}, index=pd.to_datetime(DATES))
CFG_Y_DF.index.rename(TS_VAR, inplace=True)


class TestTSConfig:

    def test_registration(self):
        cfg = TSConfig()
        cfg.register_dataset(TRAIN_DF, 'predictor', 'Timestamp', 'x')
        cfg.register_target(TARGET_DF, 'Timestamp', 'y')

        assert 'predictor' in cfg.datasets.keys() and 'target' in cfg.datasets.keys()
        pd.testing.assert_frame_equal(cfg.datasets['predictor'], CFG_X_DF)
        pd.testing.assert_frame_equal(cfg.datasets['target'], CFG_Y_DF)

    def test_registration_unspecified(self):
        cfg = TSConfig()
        cfg.register_dataset(TRAIN_DF, 'predictor', 'Timestamp')
        cfg.register_target(TARGET_DF, 'Timestamp')

        assert 'predictor' in cfg.datasets.keys() and 'target' in cfg.datasets.keys()
        pd.testing.assert_frame_equal(cfg.datasets['predictor'], CFG_X_DF)
        pd.testing.assert_frame_equal(cfg.datasets['target'], CFG_Y_DF)

    def test_registration_sorting(self):
        perm = np.array([3, 1, 2, 0, 4, 5, 6, 7])
        dates = DATES[perm]
        x = X[perm]
        y = Y[perm]

        uns_train_df = pd.DataFrame({TS_VAR: dates, 'x': x})
        uns_target_df = pd.DataFrame({TS_VAR: dates, 'y': y})

        cfg = TSConfig()
        cfg.register_dataset(uns_train_df, 'predictor', 'Timestamp', 'x')
        cfg.register_target(uns_target_df, 'Timestamp', 'y')

        pd.testing.assert_frame_equal(cfg.datasets['predictor'], CFG_X_DF)
        pd.testing.assert_frame_equal(cfg.datasets['target'], CFG_Y_DF)

    def test_add_ar(self):
        X_lag2 = np.array([np.nan, np.nan, 2, 3, 4, 4, 5, 4, 3, 1])
        dates = np.append(DATES, ['2019-02-26', '2019-03-05'])
        AR_DF = pd.DataFrame({'x_lag2': X_lag2}, index=pd.to_datetime(dates))
        AR_DF.index.rename(TS_VAR, inplace=True)

        cfg = TSConfig()
        cfg.register_dataset(TRAIN_DF, 'predictor', 'Timestamp', 'x')
        cfg.register_target(TARGET_DF, 'Timestamp', 'y')

        cfg.add_AR([2], dataset='predictor', var_names=['x'])

        pd.testing.assert_frame_equal(cfg.datasets['AR_predictor'], AR_DF)
        assert cfg.ar_set

    def test_set_delay(self):
        X_delay = np.array([np.nan, np.nan, 2, 3, 4, 4, 5, 4, 3, 1])
        dates = np.append(DATES, ['2019-02-26', '2019-03-05'])
        DELAY_DF = pd.DataFrame({'x': X_delay}, index=pd.to_datetime(dates))
        DELAY_DF.index.rename(TS_VAR, inplace=True)

        cfg = TSConfig()
        cfg.register_dataset(TRAIN_DF, 'predictor', 'Timestamp', 'x')
        cfg.register_target(TARGET_DF, 'Timestamp', 'y')

        cfg.set_delay(periods=2)
        pd.testing.assert_frame_equal(cfg.datasets['predictor'], DELAY_DF)

    def test_stack(self):
        X_new = np.array([2, 3, 4, 4, 5, 4, 3, 1, np.nan])
        y_lag = np.array([np.nan, 3, 5, 6, 6, 3, 2, 4, 1])
        dates = np.append(DATES, ['2019-02-26'])
        STACK_DF = pd.DataFrame({'x': X_new, 'y_lag1': y_lag},
                                index=pd.to_datetime(dates))
        STACK_DF.index.rename(TS_VAR, inplace=True)

        cfg = TSConfig()
        cfg.register_dataset(TRAIN_DF, 'predictor', 'Timestamp', 'x')
        cfg.register_target(TARGET_DF, 'Timestamp', 'y')

        cfg.add_AR([1], dataset='target', var_names=['y'])
        cfg.stack()

        pd.testing.assert_frame_equal(cfg.data[0], STACK_DF)
