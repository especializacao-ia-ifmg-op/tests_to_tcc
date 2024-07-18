import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
import gc

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 7

# Function definitions
def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]

def get_search_dataset_multivariate(dataset, n_var):
    #df_target = pd.read_csv('datasets/clima/lat-19.75_lon-44.45.csv', parse_dates=True, index_col='Unnamed: 0')
    #df_target = df_target.round(decimals=1)
    #df_target.fillna(0, inplace=True)
    #df_target.head()
    df1 = pd.read_csv(dataset, sep=";")
    #print(f'{df1.head()}')
    
    ints = df1.select_dtypes(include=['int64','int32','int16']).columns
    df1[ints] = df1[ints].apply(pd.to_numeric, downcast='integer')
    floats = df1.select_dtypes(include=['float']).columns
    df1[floats] = df1[floats].apply(pd.to_numeric, downcast='float')
    #print(f'{df1.head()}')
    
    series = df1.iloc[:,1:n_var+1]
    norm_df = normalize(series)
    #print(f'{df1.head()}')
    size = int(len(norm_df) * 0.80)
    train, test = norm_df[0:size], norm_df[size:len(norm_df)]
    
    return train, test

def form_data(data, t, n_execucoes, n_previsoes):
    df = pd.DataFrame(data)
    df1 = df.T
    frames = [df1.iloc[:,0], df1.iloc[:,1], df1.iloc[:,2], df1.iloc[:,3], df1.iloc[:,4], df1.iloc[:,5], df1.iloc[:,6], df1.iloc[:,7], df1.iloc[:,8], df1.iloc[:,9], df1.iloc[:,10], df1.iloc[:,11],
          df1.iloc[:,12], df1.iloc[:,13], df1.iloc[:,14], df1.iloc[:,15], df1.iloc[:,16], df1.iloc[:,17],df1.iloc[:,18], df1.iloc[:,19], df1.iloc[:,20], df1.iloc[:,21], df1.iloc[:,22], 
          df1.iloc[:,23], df1.iloc[:,24], df1.iloc[:,25], df1.iloc[:,26], df1.iloc[:,27], df1.iloc[:,28], df1.iloc[:,29]]
    result = pd.concat(frames)
    r = pd.DataFrame(result) 
    r.insert(1, "Model", True)
    for i in range(n_execucoes * n_previsoes): # n_execucoes * n_previsoes
        r['Model'].iloc[i] = 'VAR'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, n_var, n_execucoes, n_previsoes):        
    train, test = get_search_dataset_multivariate(dataset_file_name, n_var)    
    
    history = [x for x in train]
    predictions = []
    results = []
    
    for i in range(n_execucoes):
        order = 4
        for t in range(len(test)):
            model = VAR(train.values)
            model_fit = model.fit(maxlags=order)
            fc = model_fit.forecast(y=test[:t+4].values, steps=1)
            output = fc[0][6]
            yhat = output
            predictions.append(yhat)
            obs = test['ETo'].iloc[t]
            history.append(obs)
        #print(f"{test['ETo'][:5]}, {predictions[:5]}")
        
        rmse = np.sqrt(mean_squared_error(test['ETo'], predictions))
        predictions = []
        results.append(rmse)

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    del train
    del test
    del model
    del results
    gc.collect()

n_var = 7
n_execucoes=30
n_previsoes=1
print(f'[multi_all_lat-19.75_lon-44.45_VAR.py]\n')
print(f'Running...')

start = time.time()
run_model(dataset_file_name='multi_all_lat-19.75_lon-44.45.csv', result_file_name='new_m_all_results_VAR_'+str(n_execucoes)+'_'+str(n_previsoes)+'.csv', sufix=' (Rs, u2, Tmax, Tmin, RH, pr, ETo)', n_var =n_var, n_execucoes=n_execucoes, n_previsoes=n_previsoes)
stop = time.time()

print(f'...done! Execution time = {stop - start}.')
