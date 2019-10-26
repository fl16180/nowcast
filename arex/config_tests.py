import pandas as pd
import pytest

GLOBAL_DATE_VAR = 'Date'

DATES = ['2019-01-01', '2019-01-08', '2019-01-15',
         '2019-01-29', '2019-02-05', '2019-02-12',
         '2019-02-19', '2019-02-26']
TARG = [3, 5, 6, 6, 3, 2, 4, 1]
TEST_DF = pd.DataFrame({GLOBAL_DATE_VAR: DATES, 'vals': TARG})



class DataConfigTest()