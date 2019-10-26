from arex.datasets import CDCLoader, AthenaLoader
from arex.datasets import TSConfig
from arex import Arex

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



# heatmap in dataconfig to show where data missing



