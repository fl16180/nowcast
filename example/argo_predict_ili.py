""" Example: Nowcast weekly flu incidence in the state of Massachusetts """

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

import pandas as pd
from forecastlib.datasets import CDCLoader, gt_loader
from forecastlib.models import Argo
from forecastlib import TSConfig, Arex
from forecastlib.utils.metrics import RMSE, Corr


# load data
cdcl = CDCLoader('./ILI_MA.csv', ili_version='unweighted')
ili = cdcl.load_state('MA')
gt = gt_loader('./GTdata_MA.csv')

# configure data with 52 ILI lag terms
dc = TSConfig()
dc.register_dataset(ili, name='ILI', type='target')
dc.register_dataset(gt, name='Trends', type='predictor')
dc.add_AR(range(1, 53), dataset='ILI', var_names=['%ILI'])
dc.stack()

# use the standard ARGO model
base_mod = Argo()
arex = Arex(model=base_mod, data_config=dc)
out = arex.nowcast(pred_start='2017-01-01', pred_end='2018-04-22',
                   training='roll', window=104, pred_name='ARGO')

predictions = pd.merge(ili, out)
print(predictions)

print('RMSE:', RMSE(predictions['%ILI'], predictions['ARGO']))
print('Correlation:', Corr(predictions['%ILI'], predictions['ARGO']))
