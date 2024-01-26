# Imports
import Ensemble as es
import basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 7

import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

from math import sqrt

import warnings
warnings.filterwarnings("ignore")
import time
import gc

# Function definitions
def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]

def get_search_dataset_multivariate(dataset, n_var):
    df1 = pd.read_csv(dataset, sep="\s+|;|,") # 7305 registros
    
    ints = df1.select_dtypes(include=['int64','int32','int16']).columns
    df1[ints] = df1[ints].apply(pd.to_numeric, downcast='integer')
    floats = df1.select_dtypes(include=['float']).columns
    df1[floats] = df1[floats].apply(pd.to_numeric, downcast='float')
    
    series = df1.iloc[:,1:n_var+1]
    norm_df = normalize(series)
    size = int(len(norm_df) * 0.80)
    train = series[:size]
    test = series[size:]
    return train, test

def form_data(data, t, n_execucoes, n_previsoes):
    df = pd.DataFrame(data)
    df1 = df.T
    frames = [df1.iloc[:,0], df1.iloc[:,1], df1.iloc[:,2], df1.iloc[:,3], df1.iloc[:,4]]#, df1.iloc[:,5], df1.iloc[:,6], df1.iloc[:,7], df1.iloc[:,8], df1.iloc[:,9], df1.iloc[:,10], df1.iloc[:,11],
        #  df1.iloc[:,12], df1.iloc[:,13], df1.iloc[:,14], df1.iloc[:,15], df1.iloc[:,16], df1.iloc[:,17],df1.iloc[:,18], df1.iloc[:,19], df1.iloc[:,20], df1.iloc[:,21], df1.iloc[:,22], 
        #  df1.iloc[:,23], df1.iloc[:,24], df1.iloc[:,25], df1.iloc[:,26], df1.iloc[:,27], df1.iloc[:,28], df1.iloc[:,29]]
    result = pd.concat(frames)
    r = pd.DataFrame(result) 
    r.insert(1, "Model", True)
    for i in range(n_execucoes * n_previsoes): # n_execucoes * n_previsoes
        r['Model'].iloc[i] = 'CNN'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, star, n_var, n_execucoes, n_previsoes):
    train, test = get_search_dataset_multivariate(dataset_file_name, n_var=n_var)
    results = []
    train, test, scaler = es.get_dados(star, train, test)
    X_train, y_train, X_test, y_test = basic.slideWindowMulti(train, test, n_lags=star['lags'], n_var=n_var)
    for i in range(n_execucoes):
        model,_ = basic.modelo_CNN1(X_train, y_train, star)
        rmse, yhat, y_test = basic.predictModelMulti(test, model, n_previsoes=n_previsoes, n_lags=star['lags'], n_var=n_var, scaler=scaler)
        results.append(rmse)

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)
    del train
    del test
    del X_train
    del y_train
    del X_test
    del y_test
    del yhat
    del model
    del results
    gc.collect()

# Parameters
star_CNN1 = {'filters': 1, 'pool': 0, 'pool_size': 3, 'dropout': 0.012594059561340142, 'norm': 1, 'lags': 4, 'num_conv': 1, 'kernel_size': 3, 'rmse': 0.7696852129001718, 'num_param': 449}
# star_CNN2 = {'filters': 1, 'dropout': 0, 'norm': 1, 'lags': 4, 'num_conv': 1, 'kernel_size': 0, 'rmse': 0.7566198577347709, 'num_param': 449}
# star_CNN3 = {'pilhas': 2, 'filters': 1, 'dropout': 0.2, 'norm': 1, 'lags': 48, 'num_conv': 3, 'kernel_size': 2, 'rmse': 0.7530, 'num_param': 68257}

# Running the model
print(f'[multi_u2_lat-19.75_lon-44.45.py]\n')
print(f'Running...')

start = time.time()
run_model(dataset_file_name='multi_u2_lat-19.75_lon-44.45.csv', result_file_name='new_m_u2_results_CNN1_5_3.csv', sufix='1 (u2, ETo)', star=star_CNN1, n_var=2, n_execucoes=5, n_previsoes=3)
stop = time.time()

print(f'...done! Execution time = {stop - start}.')