import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import common_util as cu


def can_forecast(df, feature, target, cutoff=0.05):
    granger_results = grangercausalitytests(df[[target, feature]], maxlag=12, verbose=False)
    ssr_chi2test_results = [granger_results[key][0]['ssr_chi2test'] for key in granger_results.keys()]
    ssr_chi2test_pvalues = [result[1] for result in ssr_chi2test_results]
    pvalue = min(ssr_chi2test_pvalues)
    return pvalue < cutoff


df = pd.read_csv('data/processedMultivariateData_0.csv')

df_house = df[df['LCLid'] == 'MAC003668']
df_house, data_length = cu.data_reformat(df_house)

# visualise the data
for column in df_house.columns:
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.plot(df_house[column])
    ax.set_title(column)
    plt.tight_layout()
    plt.savefig(f'plots/original/house2_{column}.png', dpi=120)
    plt.close('all')

# feature selection
corr = spearmanr(df_house).correlation
df_corr = pd.DataFrame(corr, index=df_house.columns)
df_corr.columns = df_house.columns
distance_matrix = df_corr.applymap(lambda x: 1 - abs(x))
condensed_distances = squareform(distance_matrix, checks=False)

corr_linkage = hierarchy.ward(condensed_distances)

plt.figure(figsize=(5,6))
plt.xticks([])
plt.yticks([])
dendro = hierarchy.dendrogram(corr_linkage, labels=df_house.columns, orientation='left', leaf_font_size=12)
plt.savefig(f'plots/original/feature_selection.png', dpi=120, bbox_inches='tight')
plt.close('all')

cluster_assignments = fcluster(corr_linkage, 5, 'maxclust')
cluster_assignments = pd.Series(cluster_assignments, index=df_house.columns)
cluster_assignments.sort_values(inplace=True)
print("Cluster Assignment:\n", cluster_assignments)

# Granger's causality test
granger_results = grangercausalitytests(df[['energy', 'temperature']], maxlag=12, verbose=False)
ssr_chi2test_results = [granger_results[key][0]['ssr_chi2test'] for key in granger_results.keys()]

selected_features = [feature for feature in df_house.columns if can_forecast(df, feature, 'energy', 0.001)]
selected_features.append('energy')
print("Selected Features:\n", selected_features)