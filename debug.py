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
              training="roll", window=104)

             



# heatmap in dataconfig to show where data missing



