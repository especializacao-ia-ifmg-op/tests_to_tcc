import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

def get_search_dataset_multivariate(dataset):
    df1 = pd.read_csv(dataset, sep=";")
    
    X = df1[['Rs', 'u2', 'Tmax', 'Tmin', 'RH', 'pr']]
    y = df1['ETo']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

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
        r['Model'].iloc[i] = 'RF'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, n_execucoes, n_previsoes):
    results = []
    
    X_train_scaled, X_test_scaled, y_train, y_test = get_search_dataset_multivariate(dataset_file_name)
    
    for i in range(n_execucoes):
        #model = RandomForestRegressor(n_estimators=100, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=np.random.randint(42))
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        results.append(rmse)

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    del X_train_scaled
    del y_train
    del X_test_scaled
    del y_test
    del model
    del results
    gc.collect()

n_execucoes=30
n_previsoes=1
print(f'[multi_all_lat-19.75_lon-44.45_RF.py]\n')
print(f'Running...')

start = time.time()
run_model(dataset_file_name='multi_all_lat-19.75_lon-44.45.csv', result_file_name='new_m_all_results_RF_'+str(n_execucoes)+'_'+str(n_previsoes)+'.csv', sufix=' (Rs, u2, Tmax, Tmin, RH, pr, ETo)', n_execucoes=n_execucoes, n_previsoes=n_previsoes)
stop = time.time()

print(f'...done! Execution time = {stop - start}.')
