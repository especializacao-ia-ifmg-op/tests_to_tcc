import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# u_CNN1 = pd.read_csv('./results/u_results_CNN1.csv',delimiter=',')
# u_CNN1 = u_CNN1[u_CNN1['Unnamed: 0'].isin([0,2,6,9])]

# u_CNN1_1 = pd.read_csv('u_results_CNN1_1',delimiter=',')
# u_CNN1_1 = u_CNN1_1[u_CNN1_1['Unnamed: 0'].isin([0,2,6,9])]

m_VAR = pd.read_csv('../results/new_m_all_results_VAR_30_1.csv',delimiter=',')
m_VAR = m_VAR[m_VAR['Unnamed: 0'].isin([0,2,6,9])]

m_CNN1 = pd.read_csv('../results/new_m_all_results_CNN1_30_1.csv',delimiter=',')
m_CNN1 = m_CNN1[m_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_RF = pd.read_csv('../results/new_m_all_results_RF_30_1.csv',delimiter=',')
m_RF = m_RF[m_RF['Unnamed: 0'].isin([0,2,6,9])]

m_TimeGPT = pd.read_csv('../results/new_m_all_results_TimeGPT_30_1.csv',delimiter=',')
m_TimeGPT = m_TimeGPT[m_TimeGPT['Unnamed: 0'].isin([0,2,6,9])]

frames = [m_VAR, m_CNN1, m_RF, m_TimeGPT]#, m_u2_tmax_CNN1]
result = pd.concat(frames, ignore_index=True)
result['Unnamed: 0'] = result['Unnamed: 0'] + 1
# print(result.head(50))

# plt.style.use('seaborn')
sns.set_style("whitegrid")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[14,6], dpi=100)
g1 = sns.boxplot(x=result.iloc[:,0], y=result.iloc[:,1], hue=result.iloc[:,2], data=result, palette="tab20", linewidth=0.7, saturation=1)
plt.tick_params(labelsize=12)
# plt.xticks([1,3,7,10])
#labels = ['1','3','7','10']
labels = ['Horizonte de previsão = 1 dia']
plt.xlabel("Horizonte de previsão = 1 dia", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.savefig('boxplot_multi.png', dpi=300)
plt.show()