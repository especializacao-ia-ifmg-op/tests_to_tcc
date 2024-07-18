import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    
def to_tensor(data, features, target):
    X = torch.tensor(data[features].values, dtype=torch.float32)
    y = torch.tensor(data[target].values, dtype=torch.float32)
    return X, y    

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, src):
        # src shape: (sequence_length, batch_size, input_dim)
        transformer_out = self.transformer(src, src)
        # Take the last output (for prediction)
        out = self.fc(transformer_out[-1, :, :])
        return out

def train_model(model, X_train, y_train, num_epochs, batch_size, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            output = model(batch_X.unsqueeze(1))  # reshape to (sequence_length, batch_size, input_dim)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(X_train)}')


def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.unsqueeze(1)).squeeze()
        mse = criterion(predictions, y_test)
        return predictions, mse.item()

def get_search_dataset_multivariate(dataset):
    df1 = pd.read_csv(dataset, sep=";")
    
    features = ['Rs', 'u2', 'Tmax', 'Tmin', 'RH', 'pr']
    target = 'ETo'

    scaler = StandardScaler()
    df1[features] = scaler.fit_transform(df1[features])
    
    train_data, test_data = train_test_split(df1, test_size=0.2, random_state=42)

    X_train, y_train = to_tensor(train_data, features, target)
    X_test, y_test = to_tensor(test_data, features, target)
    input_dim = len(features)
    
    return X_train, X_test, y_train, y_test, input_dim

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
        r['Model'].iloc[i] = 'TimeGPT'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, n_execucoes, n_previsoes):
    nhead = 2
    num_layers = 2
    hidden_dim = 64
    output_dim = 1
    
    results = []
    
    X_train_scaled, X_test_scaled, y_train, y_test, input_dim = get_search_dataset_multivariate(dataset_file_name)
    
    for i in range(n_execucoes):
        
        # Inicializar o modelo
        model = TransformerModel(input_dim, nhead, num_layers, hidden_dim, output_dim)
        # Definir os parâmetros de treinamento
        learning_rate = 0.001
        num_epochs = 100
        batch_size = 32

        # Configurar o otimizador e a função de perda
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_model(model, X_train_scaled, y_train, num_epochs, batch_size, optimizer, criterion)
        
        predictions, mse = evaluate_model(model, X_test_scaled, y_test, criterion)
        
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

n_var = 7
n_execucoes=30
n_previsoes=1
print(f'[multi_all_lat-19.75_lon-44.45_TimeGPT.py]\n')
print(f'Running...')

start = time.time()
run_model(dataset_file_name='multi_all_lat-19.75_lon-44.45.csv', result_file_name='new_m_all_results_TimeGPT_'+str(n_execucoes)+'_'+str(n_previsoes)+'.csv', sufix=' (Rs, u2, Tmax, Tmin, RH, pr, ETo)', n_execucoes=n_execucoes, n_previsoes=n_previsoes)
stop = time.time()

print(f'...done! Execution time = {stop - start}.')