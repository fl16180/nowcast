from arex.datasets import CDCLoader, AthenaLoader
from arex.datasets import TSConfig
from arex import Arex
from sklearn.linear_model import LassoCV


cdcl = CDCLoader("./ILI_national_dated.csv")
cdc = cdcl.load_national()

athl = AthenaLoader("./ATHdata.csv")
ath = athl.load_national()


dc = TSConfig()
dc.register_dataset(cdc, 'CDC', 'target')
dc.register_dataset(ath, 'athena', 'predictor')
dc.add_AR(range(1, 4), dataset='CDC', var_names='%ILI')
dc.stack()

dc.data


base_mod = LassoCV(cv=10)
m = Arex(model=base_mod, data_config=dc)

m.predict(pred_start="2019-04-07", pred_end="2019-04-28",
              training="expand", window=10)

             


DATES = np.array(['2019-01-01', '2019-01-08', '2019-01-15',
                  '2019-01-22', '2019-01-29', '2019-02-05',
                  '2019-02-12', '2019-02-19'])
X = np.array([3.1, 5.1, 6.1, 6.1, 3.1, 2.1, 4.1, 1.1])
Y = np.array([3, 5, 6, 6, 3, 2, 4, 1])

TRAIN_DF = pd.DataFrame({TS_VAR: DATES, 'x': X})
TARGET_DF = pd.DataFrame({TS_VAR: DATES, 'y': Y})

class MockModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return [1]

c3 = TSConfig()
c3.register_dataset(TARGET_DF, 'y', 'target')
c3.register_dataset(TRAIN_DF, 'x', 'predictor')
c3.set_delay(1)
# c3.add_AR([1], dataset='x')

# c3.add_AR([1], dataset='y')
c3.stack()
c3.data[0]
c3.data[1]

mm = MockModel()
m = Arex(model=mm, data_config=c3)

m.predict(pred_start="2019-02-12", pred_end="2019-03-05",
          training="roll", window=2)



# heatmap in dataconfig to show where data missing



