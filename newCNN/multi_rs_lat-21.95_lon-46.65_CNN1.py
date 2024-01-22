import Ensemble as es
import basic
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import warnings
# from google.colab import files
warnings.filterwarnings("ignore")

def get_search_dataset_multivariate(dataset, n_var):
    df1 = pd.read_csv(dataset, sep=',') # 7305 registros
    series = df1.iloc[:,1:n_var+1]
    train = series[:6939]
    test = series[6939:]
    return train, test

def form_data(data, t):  
	df = pd.DataFrame(data)
	df1 = df.T
	frames = [df1.iloc[:,0], df1.iloc[:,1], df1.iloc[:,2], df1.iloc[:,3], df1.iloc[:,4]]#, df1.iloc[:,5], df1.iloc[:,6], df1.iloc[:,7], df1.iloc[:,8], df1.iloc[:,9], df1.iloc[:,10], df1.iloc[:,11],
        #  df1.iloc[:,12], df1.iloc[:,13], df1.iloc[:,14], df1.iloc[:,15], df1.iloc[:,16], df1.iloc[:,17],df1.iloc[:,18], df1.iloc[:,19], df1.iloc[:,20], df1.iloc[:,21], df1.iloc[:,22], 
        #  df1.iloc[:,23], df1.iloc[:,24], df1.iloc[:,25], df1.iloc[:,26], df1.iloc[:,27], df1.iloc[:,28], df1.iloc[:,29]]
	result = pd.concat(frames)
	r = pd.DataFrame(result) 
	r.insert(1, "Model", True) 
	for i in range(50):
	  r['Model'].iloc[i] = 'CNN'+ t
	return r

star_CNN1 = {'filters': 1, 'pool': 0, 'pool_size': 3, 'dropout': 0.012594059561340142, 'norm': 1, 'lags': 4, 'num_conv': 1, 'kernel_size': 3, 'rmse': 0.7696852129001718, 'num_param': 449}

m_row_train, m_row_test = get_search_dataset_multivariate('multi_rs_lat-21.95_lon-46.65.csv', n_var=2)

m_rs_results_CNN1 = []
n_var = 2
m_train, m_test, m_scaler = es.get_dados(star_CNN1, m_row_train, m_row_test)
X_train, y_train, X_test, y_test = basic.slideWindowMulti(m_train, m_test, n_lags=star_CNN1['lags'], n_var=n_var)
for i in range(5):
	model,_ = basic.modelo_CNN1(X_train, y_train, star_CNN1) 
	rmse, yhat, y_test = basic.predictModelMulti(m_test, model, n_previsoes=10, n_lags=star_CNN1['lags'], n_var=n_var, scaler=m_scaler)
	m_rs_results_CNN1.append(rmse)

m_rs_results_CNN1 = form_data(m_rs_results_CNN1, '1 (Rs, ETo)')
m_rs_results_CNN1.to_csv('m_rs_results_CNN1.csv',index=True)