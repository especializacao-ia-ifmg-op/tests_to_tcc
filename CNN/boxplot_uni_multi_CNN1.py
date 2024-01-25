import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

u_CNN1 = pd.read_csv('u_results_CNN1.csv',delimiter=',')
u_CNN1 = u_CNN1[u_CNN1['Unnamed: 0'].isin([0,2,6,9])]

# u_CNN1_1 = pd.read_csv('u_results_CNN1_1',delimiter=',')
# u_CNN1_1 = u_CNN1_1[u_CNN1_1['Unnamed: 0'].isin([0,2,6,9])]

m_rs_CNN1 = pd.read_csv('m_rs_results_CNN1.csv',delimiter=',')
m_rs_CNN1 = m_rs_CNN1[m_rs_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_u2_CNN1 = pd.read_csv('m_u2_results_CNN1.csv',delimiter=',')
m_u2_CNN1 = m_u2_CNN1[m_u2_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_rh_CNN1 = pd.read_csv('m_rh_results_CNN1.csv',delimiter=',')
m_rh_CNN1 = m_rh_CNN1[m_rh_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_tmax_CNN1 = pd.read_csv('m_tmax_results_CNN1.csv',delimiter=',')
m_tmax_CNN1 = m_tmax_CNN1[m_tmax_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_tmin_CNN1 = pd.read_csv('m_tmin_results_CNN1.csv',delimiter=',')
m_tmin_CNN1 = m_tmin_CNN1[m_tmin_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_pr_CNN1 = pd.read_csv('m_pr_results_CNN1.csv',delimiter=',')
m_pr_CNN1 = m_pr_CNN1[m_pr_CNN1['Unnamed: 0'].isin([0,2,6,9])]

m_rs_u2_CNN1 = pd.read_csv('m_rs_u2_results_CNN1.csv',delimiter=',')
m_rs_u2_CNN1 = m_rs_u2_CNN1[m_rs_u2_CNN1['Unnamed: 0'].isin([0,2,6,9])]

frames = [u_CNN1, m_rs_CNN1, m_u2_CNN1, m_rh_CNN1, m_tmax_CNN1, m_tmin_CNN1, m_pr_CNN1, m_rs_u2_CNN1]
result = pd.concat(frames, ignore_index=True)
result['Unnamed: 0'] = result['Unnamed: 0'] + 1
print(result.head(50))

# plt.style.use('seaborn')
sns.set_style("whitegrid")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[14,6], dpi=100)
g1 = sns.boxplot(x=result.iloc[:,0], y=result.iloc[:,1], hue=result.iloc[:,2], data=result, palette="tab20", linewidth=0.7, saturation=1)
plt.tick_params(labelsize=12)
# plt.xticks([1,3,7,10])
labels = ['1','3','7','10']
plt.xlabel("Horizontes de previsão", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.savefig('boxplot_uni_multi.png', dpi=300)
plt.show()