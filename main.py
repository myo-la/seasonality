import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

tint_type = "dark_tints"
# tint_type = "polarised"

data_orig = pd.read_csv('CSVs/' + tint_type + '.csv', encoding='ISO-8859-1')
data_orig['month'] = pd.to_datetime(data_orig['month'], format='%d/%m/%Y')
data_orig = data_orig[data_orig['month'] < "2022-01-01"]
# data_orig = data_orig[data_orig['month'] > "2016-12-31"]

data_orig.set_index('month', inplace=True)

analysis = data_orig[[tint_type]].copy()

decompose_result_mult = seasonal_decompose(analysis, model="multiplicative")

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid


decomp_df = pd.DataFrame({'month': data_orig.index, 'trend': trend, 'seasonal': seasonal, 'residual': residual})

decomp_df.to_csv(r'CSVs\trend_' + tint_type + '.csv', index=False)

decompose_result_mult.plot()
plt.show()
