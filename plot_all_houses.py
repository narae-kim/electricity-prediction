import pandas as pd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import common_util as cu

df = pd.read_csv('data/processedMultivariateData_0.csv')
houses = ['3422', '3668', '3737', '3851']
num = 1
for house in houses:
    df_house = df[df['LCLid'] == 'MAC00' + house]
    df_house, data_length = cu.preprocess(df_house)
    fig = plt.figure(figsize=[8, 3])
    ax = fig.add_subplot(111)
    plt.plot(df_house.index, df_house['energy'], linewidth=0.2, label=f'house {num}')
    plt.ylabel('Electricity Usage (kWh)', fontsize=11)
    plt.legend(loc='best', fontsize=11)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
    plt.savefig(f'plots/original/original_graph_{num}.png')
    plt.close('all')
    num += 1