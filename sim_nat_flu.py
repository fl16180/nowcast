from forecastlib.datasets import CDCLoader, AthenaLoader
from forecastlib.models import Argo2015, ArgoSVM
from forecastlib import TSConfig, Arex


cdcl = CDCLoader("./ILI_national_dated.csv", ili_version='weighted')
cdc = cdcl.load_national()

athl = AthenaLoader("./ATHdata.csv")
ath = athl.load_national()

dc = TSConfig()
dc.register_dataset(cdc, 'CDC', 'target')
dc.register_dataset(ath, 'athena', 'predictor')
dc.add_AR(range(1, 53), dataset='CDC', var_names=['%ILI'])
dc.stack()


lasso_mod = Argo2015()
svm_mod = ArgoSVM()

arex = Arex(model=lasso_mod, data_config=dc)
p1 = arex.nowcast(pred_start='2019-10-20', pred_end='2019-11-17',
                  training='roll', window=104, pred_name='ARGO 1wk')

arex = Arex(model=svm_mod, data_config=dc)
p2 = arex.forecast(t_plus=1, pred_start='2019-10-13', pred_end='2019-11-17',
                   training='roll', window=104, pred_name='ARGO 2wk')

arex = Arex(model=svm_mod, data_config=dc)
p3 = arex.forecast(t_plus=2, pred_start='2019-10-06', pred_end='2019-11-17',
                   training='roll', window=104, pred_name='ARGO 3wk')

arex = Arex(model=lasso_mod, data_config=dc)
p4 = arex.forecast(t_plus=3, pred_start='2019-09-29', pred_end='2019-11-17',
                   training='roll', window=104, pred_name='ARGO 4wk')

print(p1)
ll = arex.get_log()
for i in ll:
    print(i)