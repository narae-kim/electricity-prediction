import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import common_util as cu


df = pd.read_csv('data/processedMultivariateData_0.csv')

df_house = df[df['LCLid'] == 'MAC003668']
df_house, data_length = cu.data_reformat(df_house)

# seasonality
fig = plot_acf(df_house['energy'], lags=336)
fig.set_size_inches(9,3)
plt.title("Autocorrelation of energy")
plt.savefig(f'plots/original/autocorrelation_336.png')

# removing seasonality
df_house['lag-48'] = df_house['energy'] - df_house['energy'].shift(48)

fig = plot_acf(df_house['lag-48'].dropna(), lags=100, zero=False)
fig.set_size_inches(9,4)
plt.title("Autocorrelation of energy after removing seasonality with lag 48")
plt.savefig(f'plots/original/autocorrelation_removing_seasonality.png', dpi=120)
