import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from arex.datasets.data_config import TSConfig, TS_VAR
from arex import Arex


class MockModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return [1]


def to_ymd(datetime):
    return datetime.strftime('%Y-%m-%d')


DATES = np.array(['2019-01-01', '2019-01-08', '2019-01-15',
                  '2019-01-22', '2019-01-29', '2019-02-05',
                  '2019-02-12', '2019-02-19'])
X = np.array([3.1, 5.1, 6.1, 6.1, 3.1, 2.1, 4.1, 1.1])
Y = np.array([3, 5, 6, 6, 3, 2, 4, 1])

TRAIN_DF = pd.DataFrame({TS_VAR: DATES, 'x': X})
TARGET_DF = pd.DataFrame({TS_VAR: DATES, 'y': Y})

mm = MockModel()


class TestArexNowcast:
    def test_simple_nowcast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.nowcast(pred_start="2019-02-12", pred_end="2019-02-19",
                    training="roll", window=2)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-12'
        assert to_ymd(log[1]['time']) == '2019-02-19'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-01-29'
        assert to_ymd(info['X_train'][1]) == '2019-02-05'
        assert to_ymd(info['y_train'][0]) == '2019-01-29'
        assert to_ymd(info['y_train'][1]) == '2019-02-05'
        assert to_ymd(info['X_pred']) == '2019-02-12'
        assert info['sizes'][0] == (2, 1)

    def test_delay_nowcast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.set_delay(1, datasets=['x'])
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.nowcast(pred_start="2019-02-19", pred_end="2019-02-26",
                    training="roll", window=2)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-19'
        assert to_ymd(log[1]['time']) == '2019-02-26'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-02-05'
        assert to_ymd(info['X_train'][1]) == '2019-02-12'
        assert to_ymd(info['y_train'][0]) == '2019-02-05'
        assert to_ymd(info['y_train'][1]) == '2019-02-12'
        assert to_ymd(info['X_pred']) == '2019-02-19'
        assert info['sizes'][0] == (2, 1)

    def test_ar_nowcast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.add_AR([2], dataset='x')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.nowcast(pred_start="2019-02-19", pred_end="2019-03-05",
                    training="roll", window=2)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-19'
        assert to_ymd(log[1]['time']) == '2019-02-26'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-02-05'
        assert to_ymd(info['X_train'][1]) == '2019-02-12'
        assert to_ymd(info['y_train'][0]) == '2019-02-05'
        assert to_ymd(info['y_train'][1]) == '2019-02-12'
        assert to_ymd(info['X_pred']) == '2019-02-19'
        assert info['sizes'][0] == (2, 2)

        info = log[1]
        assert to_ymd(info['X_train'][0]) == '2019-02-12'
        assert to_ymd(info['X_train'][1]) == '2019-02-19'
        assert to_ymd(info['y_train'][0]) == '2019-02-12'
        assert to_ymd(info['y_train'][1]) == '2019-02-19'
        assert to_ymd(info['X_pred']) == '2019-02-26'
        assert info['sizes'][0] == (2, 2)

        info = log[2]
        assert to_ymd(info['X_train'][0]) == '2019-02-12'
        assert to_ymd(info['X_train'][1]) == '2019-02-19'
        assert to_ymd(info['y_train'][0]) == '2019-02-12'
        assert to_ymd(info['y_train'][1]) == '2019-02-19'
        assert to_ymd(info['X_pred']) == '2019-03-05'
        assert info['sizes'][0] == (2, 2)

    def test_window_too_big(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.add_AR([2], dataset='x')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        with pytest.raises(ValueError):
            arex.nowcast(pred_start="2019-02-19", pred_end="2019-03-05",
                        training="expand", window=8)

        arex.nowcast(pred_start="2019-02-19", pred_end="2019-03-05",
            training="expand", window=7)
        log = arex.get_log()

        assert to_ymd(log[0]['X_train'][0]) == '2019-01-01'


class TestArexForecast:
    def test_simple_forecast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.forecast(t_plus=2, pred_start="2019-02-12", pred_end="2019-02-19",
                      training="roll", window=2, t_known=False)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-12'
        assert to_ymd(log[1]['time']) == '2019-02-19'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-01-15'
        assert to_ymd(info['X_train'][1]) == '2019-01-22'
        assert to_ymd(info['y_train'][0]) == '2019-01-29'
        assert to_ymd(info['y_train'][1]) == '2019-02-05'
        assert to_ymd(info['X_pred']) == '2019-02-12'
        assert info['sizes'][0] == (2, 1)

    def test_t_known_forecast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.forecast(t_plus=2, pred_start="2019-02-12", pred_end="2019-02-19",
                      training="roll", window=2, t_known=True)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-12'
        assert to_ymd(log[1]['time']) == '2019-02-19'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-01-22'
        assert to_ymd(info['X_train'][1]) == '2019-01-29'
        assert to_ymd(info['y_train'][0]) == '2019-02-05'
        assert to_ymd(info['y_train'][1]) == '2019-02-12'
        assert to_ymd(info['X_pred']) == '2019-02-12'
        assert info['sizes'][0] == (2, 1)

    def test_expand_forecast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.forecast(t_plus=2, pred_start="2019-02-12", pred_end="2019-02-19",
                      training="expand", window=2, t_known=False)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-12'
        assert to_ymd(log[1]['time']) == '2019-02-19'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-01-15'
        assert to_ymd(info['X_train'][1]) == '2019-01-22'
        assert to_ymd(info['y_train'][0]) == '2019-01-29'
        assert to_ymd(info['y_train'][1]) == '2019-02-05'
        assert to_ymd(info['X_pred']) == '2019-02-12'
        assert info['sizes'][0] == (2, 1)

        info = log[1]
        assert to_ymd(info['X_train'][0]) == '2019-01-15'
        assert to_ymd(info['X_train'][1]) == '2019-01-29'
        assert to_ymd(info['y_train'][0]) == '2019-01-29'
        assert to_ymd(info['y_train'][1]) == '2019-02-12'
        assert to_ymd(info['X_pred']) == '2019-02-19'
        assert info['sizes'][0] == (3, 1)

    def test_t_known_expand_forecast(self):
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.forecast(t_plus=2, pred_start="2019-02-12", pred_end="2019-02-19",
                      training="expand", window=2, t_known=True)
        log = arex.get_log()
        assert to_ymd(log[0]['time']) == '2019-02-12'
        assert to_ymd(log[1]['time']) == '2019-02-19'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-01-22'
        assert to_ymd(info['X_train'][1]) == '2019-01-29'
        assert to_ymd(info['y_train'][0]) == '2019-02-05'
        assert to_ymd(info['y_train'][1]) == '2019-02-12'
        assert to_ymd(info['X_pred']) == '2019-02-12'
        assert info['sizes'][0] == (2, 1)

        info = log[1]
        assert to_ymd(info['X_train'][0]) == '2019-01-22'
        assert to_ymd(info['X_train'][1]) == '2019-02-05'
        assert to_ymd(info['y_train'][0]) == '2019-02-05'
        assert to_ymd(info['y_train'][1]) == '2019-02-19'
        assert to_ymd(info['X_pred']) == '2019-02-19'
        assert info['sizes'][0] == (3, 1)

    def test_ar_forecast(self):
        """ using AR to extend X to be longer than Y so must train earlier """
        cfg = TSConfig()
        cfg.register_dataset(TARGET_DF, 'y', 'target')
        cfg.register_dataset(TRAIN_DF, 'x', 'predictor')
        cfg.add_AR([2], dataset='x')
        cfg.stack()

        arex = Arex(model=mm, data_config=cfg)
        arex.forecast(t_plus=2, pred_start="2019-02-26", pred_end="2019-03-05",
                      training="roll", window=2, t_known=False)
        log = arex.get_log()

        assert to_ymd(log[0]['time']) == '2019-02-26'
        assert to_ymd(log[1]['time']) == '2019-03-05'

        info = log[0]
        assert to_ymd(info['X_train'][0]) == '2019-01-29'
        assert to_ymd(info['X_train'][1]) == '2019-02-05'
        assert to_ymd(info['y_train'][0]) == '2019-02-12'
        assert to_ymd(info['y_train'][1]) == '2019-02-19'
        assert to_ymd(info['X_pred']) == '2019-02-26'
        assert info['sizes'][0] == (2, 2)

        info = log[1]
        assert to_ymd(info['X_train'][0]) == '2019-01-29'
        assert to_ymd(info['X_train'][1]) == '2019-02-05'
        assert to_ymd(info['y_train'][0]) == '2019-02-12'
        assert to_ymd(info['y_train'][1]) == '2019-02-19'
        assert to_ymd(info['X_pred']) == '2019-03-05'
        assert info['sizes'][0] == (2, 2)
