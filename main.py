import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch import Tensor 
from sklearn.preprocessing import MinMaxScaler
import os
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose 

device = "cuda" if torch.cuda.is_available() else "cpu"

#Gael
path_to_csv = 'dataKCx.csv'

#Sophia
#path_to_csv = 'drive/MyDrive/kepler_test/dataKCx.csv'

#FNN, LSTM2, LSTM_multifeatures, LSTM3, LSTM4input, EXPERIMENT
#model_to_run = 'FNN'
#batch_size = 100
#num_epochs = 20

class MinMaxNormalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, tensor):
        self.min_val = tensor.min().values[0]
        self.max_val = tensor.max().values[0]

    def normalize(self, tensor):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call 'fit' first to compute min-max values.")
        return (tensor - self.min_val) / (self.max_val - self.min_val)

    def fit_normalize(self, tensor):
        self.fit(tensor)
        return self.normalize(tensor)

    def denormalize(self, normalized_tensor):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call 'fit' first to compute min-max values.")
        return normalized_tensor * (self.max_val - self.min_val) + self.min_val

class VolumeDataset(Dataset):
    def __init__(self, path_to_csv, Sn=104*3 + 1, dataset='train', model_name = 'LSTM'):
        '''
        Sn: how many days back to look for the continuous or opening volumes
        dataset: train, validation or test
        '''

        self.data = pd.read_csv(path_to_csv)
        self.Sn = Sn
        #self.train = train
        self.dataset = dataset
        self.model_name = model_name

        self.data.rename(columns = {"Unnamed: 0":"id"}, inplace=True)

        #add dummy timesamp for opening and closing auction
        opening_time = pd.Timestamp('06:55:00').strftime("%H:%M:%S")
        self.data.loc[self.data['tradingphase'] == 'OPENING AUCTION','time'] = opening_time
        closing_time = pd.Timestamp('15:30:00').strftime("%H:%M:%S")
        self.data.loc[self.data['tradingphase'] == 'CLOSING AUCTION','time'] = closing_time
        self.data['date_time'] = pd.to_datetime(self.data['date'] + 'T' + self.data['time'])
        self.data.set_index('date_time')

        #period k and date d integers used in final performance evaluation
        self.data['k'] = (self.data['id']-1)%104 + 1
        self.data['d'] = (self.data['id'] -1)//104 + 1
        #pd.get_dummies(self.data.dw).rename({'0':'dw'})

        #compute total volume
        self.total_volume_per_day = self.data.groupby('date')['volume'].sum()
        self.data = self.data.merge(self.total_volume_per_day, on='date', how='left')
        self.data.rename(columns = {"volume_x":"volume", "volume_y":"total_volume"}, inplace=True)

        ##compute total volume norm
        #self.total_volume_per_day_norm = self.total_volume_per_day/168000000
        #self.data = self.data.merge(self.total_volume_per_day_norm, on='date', how='left')
        #self.data.rename(columns = {"volume_x":"volume", "volume_y":"total_volume_norm"}, inplace=True)

        #compute cumulated sum
        self.data['cum_volume']  = self.data['volume'].cumsum()

        #compute missing intraday
        self.data['missing_intraday'] = 104 - self.data['k']
        #compute week 1-hot
        self.data['m'] = self.data['date_time'].dt.month
        self.data['dw'] = self.data['date_time'].dt.dayofweek
        self.data = pd.concat([self.data, pd.get_dummies(self.data['m'])], axis=1)
        self.data.rename(columns = {1:"m1", 2:"m2", 3:"m3", 4:"m4", 5:"m5", 6:"m6", 7:"m7", 8:"m8", 9:"m9", 10:"m10", 11:"m11", 12:"m12"}, inplace=True)
        self.data = pd.concat([self.data, pd.get_dummies(self.data['dw'])], axis=1)
        self.data.rename(columns = {0:"dw0", 1:"dw1", 2:"dw2", 3:"dw3", 4:"dw4"}, inplace=True)

        ##
        #Seasonal decomposition
        decomposed = seasonal_decompose(self.data['volume'],  model ='additive', period = 104)
        self.data['volume_sd_day_trend'] = decomposed.trend.iloc[:]
        self.data['volume_sd_day_seasonal'] = decomposed.seasonal.iloc[:]
        decomposed = seasonal_decompose(self.data['volume'],  model ='additive', period = 104*5)
        self.data['volume_sd_week_trend'] = decomposed.trend.iloc[:]
        self.data['volume_sd_week_seasonal'] = decomposed.seasonal.iloc[:]


        #separated data in train, validation, test
        #use about 20% for validation
        end_train_day = 348 #2021-10-08
        end_valid_day = 438 #2022-03-08
        self.train_dataset = self.data[self.data['d'] <= end_train_day]
        self.validation_dataset = self.data[(self.data['d'] > end_train_day) & (self.data['d'] <= end_valid_day)]
        self.test_dataset = self.data[(self.data['d'] > end_valid_day)]
        #last_training_date_dt = dt.datetime.strptime(last_training_date, '%Y-%m-%d').date()
        #self.train_dataset = self.data[self.data['date_time'].dt.date <= last_training_date_dt]
        #self.test_dataset = self.data[self.data['date_time'].dt.date > last_training_date_dt]
        #self.test_dataset = self.data[self.data['date'] > last_training_date]
        #self.train_dataset = self.data[self.data['date'] <= last_training_date]
        #self.test_dataset = self.data[self.data['date'] > last_training_date]

        ####
        ##Normalization
        ####
        self.scaler= MinMaxScaler(feature_range=(0, 1))
        self.train_dataset.loc[:, 'volume'] = self.scaler.fit_transform(self.train_dataset[['volume']])
        self.validation_dataset.loc[:, 'volume'] = self.scaler.transform(self.validation_dataset[['volume']])
        self.test_dataset.loc[:, 'volume'] = self.scaler.transform(self.test_dataset[['volume']])
        #use same scaler for seasonal decomposition
        #use same scaler for seasonal decomposition
        self.normalizer = MinMaxNormalizer()
        self.normalizer.fit(self.train_dataset[['volume']])
        self.train_dataset.loc[:, 'volume_sd_day_trend'] = self.normalizer.normalize(self.train_dataset[['volume_sd_day_trend']])
        self.validation_dataset.loc[:, 'volume_sd_day_trend'] = self.normalizer.normalize(self.validation_dataset[['volume_sd_day_trend']])
        self.test_dataset.loc[:, 'volume_sd_day_trend'] = self.normalizer.normalize(self.test_dataset[['volume_sd_day_trend']])
        self.train_dataset.loc[:, 'volume_sd_day_seasonal'] = self.normalizer.normalize(self.train_dataset[['volume_sd_day_seasonal']])
        self.validation_dataset.loc[:, 'volume_sd_day_seasonal'] = self.normalizer.normalize(self.validation_dataset[['volume_sd_day_seasonal']])
        self.test_dataset.loc[:, 'volume_sd_day_seasonal'] = self.normalizer.normalize(self.test_dataset[['volume_sd_day_seasonal']])
        #
        self.train_dataset.loc[:, 'volume_sd_week_trend'] = self.normalizer.normalize(self.train_dataset[['volume_sd_week_trend']])
        self.validation_dataset.loc[:, 'volume_sd_week_trend'] = self.normalizer.normalize(self.validation_dataset[['volume_sd_week_trend']])
        self.test_dataset.loc[:, 'volume_sd_week_trend'] = self.normalizer.normalize(self.test_dataset[['volume_sd_week_trend']])
        self.train_dataset.loc[:, 'volume_sd_week_seasonal'] = self.normalizer.normalize(self.train_dataset[['volume_sd_week_seasonal']])
        self.validation_dataset.loc[:, 'volume_sd_week_seasonal'] = self.normalizer.normalize(self.validation_dataset[['volume_sd_week_seasonal']])
        self.test_dataset.loc[:, 'volume_sd_week_seasonal'] = self.normalizer.normalize(self.test_dataset[['volume_sd_week_seasonal']])


        self.total_volume_normalizer = MinMaxNormalizer()
        self.train_dataset.loc[:, 'total_volume'] = self.total_volume_normalizer.fit_normalize(self.train_dataset[['total_volume']])
        self.validation_dataset.loc[:, 'total_volume'] = self.total_volume_normalizer.normalize(self.validation_dataset[['total_volume']])
        self.test_dataset.loc[:, 'total_volume'] = self.total_volume_normalizer.normalize(self.test_dataset[['total_volume']])

        self.cum_volume_normalizer = MinMaxNormalizer()
        self.train_dataset.loc[:, 'cum_volume'] = self.cum_volume_normalizer.fit_normalize(self.train_dataset[['cum_volume']])
        self.validation_dataset.loc[:, 'cum_volume'] = self.cum_volume_normalizer.normalize(self.validation_dataset[['cum_volume']])
        self.test_dataset.loc[:, 'cum_volume'] = self.cum_volume_normalizer.normalize(self.test_dataset[['cum_volume']])


    def __len__(self):
        assert self.dataset in ['train', 'validation', 'test'], 'Wrong dataset parameter'
        if self.dataset == 'train':
            return len(self.train_dataset) - self.Sn + 1
        elif self.dataset == 'validation':
            return len(self.validation_dataset) - self.Sn + 1
        elif self.dataset == 'test':
            return len(self.test_dataset) - self.Sn + 1

    def __getitem__(self, idx):
        if self.dataset == 'train':
            output_dataset = self.train_dataset[idx:idx + self.Sn]
        elif self.dataset == 'validation':
            output_dataset = self.validation_dataset[idx:idx + self.Sn]
        elif self.dataset == 'test':
            output_dataset = self.test_dataset[idx:idx + self.Sn]

        Sn = self.Sn

        if self.model_name == 'FNN' or self.model_name == 'LSTM3':
            volume = output_dataset['volume']
            total_volume = output_dataset['total_volume']
            missing_intraday = output_dataset['missing_intraday']
            return {
                'volume' : torch.tensor(volume.values, dtype=torch.float32), 
                'total_volume' : torch.tensor(total_volume.values, dtype=torch.float32), 
                'missing_intraday' : torch.tensor(missing_intraday.values, dtype=torch.float32), 
            }
        elif self.model_name == 'LSTM2':
            volume = output_dataset['volume']
            total_volume = output_dataset['total_volume']
            missing_intraday = output_dataset['missing_intraday']
            return {
                'volume' : torch.tensor(volume.values, dtype=torch.float32), 
                'total_volume' : torch.tensor(total_volume.values, dtype=torch.float32), 
                'missing_intraday' : torch.tensor(missing_intraday.values, dtype=torch.float32), 
            }
        
        elif self.model_name == 'LSTM4input':
            missing_intraday = 104 - (idx + 1)%104
            output_dataset.insert(0, 'periods_to_end', range(Sn + missing_intraday, missing_intraday, -1))
            periods_to_end = output_dataset['periods_to_end']/(104*Sn)
            volume = output_dataset['volume']
            total_volume = output_dataset['total_volume']
            week_day = output_dataset['dw']
            month = output_dataset['m']
            return {
                'volume' : torch.tensor(volume.values, dtype=torch.float32), 
                'total_volume' : torch.tensor(total_volume.values, dtype=torch.float32), 
                'periods_to_end' : torch.tensor(periods_to_end.values, dtype=torch.float32), 
                'week_day' : torch.tensor(week_day.values, dtype=torch.float32),
                'month' : torch.tensor(month.values, dtype=torch.float32)
            }
        
        elif self.model_name == "LSTM_conv_volume_decomp_periodsToEnd":
            missing_intraday = 104 - (idx + 1)%104
            output_dataset.insert(0, 'periods_to_end', range(Sn + missing_intraday, missing_intraday, -1))
            periods_to_end = output_dataset['periods_to_end']/(104*Sn)
            total_volume = output_dataset['total_volume']
            #cum_volume = output_dataset['cum_volume']
            volume = output_dataset['volume']
            return {
                'volume' : torch.tensor(volume.values, dtype=torch.float32),
                'total_volume' : torch.tensor(total_volume.values, dtype=torch.float32),
                'periods_to_end' : torch.tensor(periods_to_end.values, dtype=torch.float32),
                'volume_sd_day_trend' : torch.tensor(output_dataset['volume_sd_day_trend'].values, dtype=torch.float32),
                'volume_sd_day_seasonal' : torch.tensor(output_dataset['volume_sd_day_seasonal'].values,dtype=torch.float32),
                'volume_sd_week_trend' : torch.tensor(output_dataset['volume_sd_week_trend'].values, dtype=torch.float32),
                'volume_sd_week_seasonal' : torch.tensor(output_dataset['volume_sd_week_seasonal'].values, dtype=torch.float32)
            }

        elif self.model_name == 'EXPERIMENT':
            #add periods to end
            missing_intraday = 104 - (idx + 1)%104
            output_dataset.insert(0, 'periods_to_end', range(Sn + missing_intraday, missing_intraday, -1))
            #CONTINUOUS
            #output_dataset_close = output_dataset[output_dataset['k'] == 104]
            #output_dataset = output_dataset[output_dataset['k'] != 104]
            periods_to_end = output_dataset['periods_to_end']/(104*Sn)
            volume = output_dataset['volume']
            total_volume = output_dataset['total_volume']
            week_day = output_dataset['dw']
            month = output_dataset['m']
            #CLOSING DATA
            #volume_close = output_dataset_close['volume']
            return {
                'volume' : torch.tensor(volume.values, dtype=torch.float32), 
                'total_volume' : torch.tensor(total_volume.values, dtype=torch.float32), 
                'periods_to_end' : torch.tensor(periods_to_end.values, dtype=torch.float32), 
                'week_day' : torch.tensor(week_day.values, dtype=torch.float32),
                'dw_one_hot': torch.tensor(output_dataset[['dw0','dw1','dw2','dw3','dw4']].values, dtype=torch.float32),
                'm_one_hot': torch.tensor(output_dataset[['m1','m2','m3','m4','m5','m6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']].values, dtype=torch.float32),
                #'volume_close':torch.tensor(volume_close.values, dtype=torch.float32)
            }

def train_and_eval_model(model=None, 
                         input_path='dataKCx.csv', 
                         model_name='my_model', 
                         Sn = 3*104 + 1, 
                         optimizer=None, 
                         batch_size=50, 
                         num_epochs=10,
                         version=0,
                         save_plots=False, 
                         save_model=False,
                         num_workers=1):
    #define loss and performance metric
    class RMSE(nn.Module):
        def forward(self, y_pred, y_true):
            squared_error = (y_pred - y_true) ** 2
            ratio = squared_error/y_true**2
            return torch.sqrt(ratio.mean())

    criterion = nn.MSELoss()
    rmse = RMSE()
    
    #get datasets
    train = VolumeDataset(path_to_csv, Sn=Sn, dataset='train', model_name=model_name)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test = VolumeDataset(path_to_csv, Sn=Sn, dataset='validation', model_name=model_name)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    denorm = train.total_volume_normalizer.denormalize

    ###
    #TRAINING
    ###
    
    def train(data_loader, model, loss_fn, optimizer):
        model.train()
        loss_list = []
        rmse_loss_list = []
        #import time
        #start = time.time()
        for batch in data_loader:
            prediction = model(model.prepare_input(batch))
            target = model.prepare_target(batch)
            loss = loss_fn(prediction, target)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rmse_loss = rmse(denorm(prediction), denorm(target))
                rmse_loss_list.append(rmse_loss.item())
        average_loss = sum(loss_list)/len(loss_list)
        average_rmse_loss = sum(rmse_loss_list)/len(rmse_loss_list)
        return average_loss, average_rmse_loss
    
    def test(data_loader, model, loss_fn):
        model.eval()
        loss_list = []
        rmse_loss_list = []
        with torch.no_grad():
            for batch in data_loader:
                prediction = model(model.prepare_input(batch))
                target = model.prepare_target(batch)
                loss = loss_fn(prediction, target)
                loss_list.append(loss.item())
                rmse_loss = rmse(denorm(prediction), denorm(target))
                rmse_loss_list.append(rmse_loss.item())
            average_loss = sum(loss_list)/len(loss_list)
            average_rmse_loss = sum(rmse_loss_list)/len(rmse_loss_list)
            return average_loss, average_rmse_loss
    
    loss_list_train = []
    rmse_loss_list_train = []
    loss_list_test = []
    rmse_loss_list_test = []
    for epoch in range(num_epochs):
        average_loss_train, average_rmse_loss_train = train(train_loader, model, criterion, optimizer)
        average_loss_test, average_rmse_loss_test = test(test_loader, model, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_loss_train}, Validation Loss: {average_loss_test}, RMSE Train Loss: {average_rmse_loss_train}, RMSE Validation Loss: {average_rmse_loss_test}")
        loss_list_train.append(average_loss_train)
        rmse_loss_list_train.append(average_rmse_loss_train)
        loss_list_test.append(average_loss_test)
        rmse_loss_list_test.append(average_rmse_loss_test)
    
    #do plot
    epochs = range(1, num_epochs + 1)
    plt.figure(0)
    plt.plot(epochs, loss_list_train, label='Training Loss', color='b')
    plt.plot(epochs, loss_list_test, label='Validation Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    working_directory = os.path.dirname(os.path.realpath(__file__))
    if save_plots:
        plt.savefig(f'MSE_loss_{model_name}_{version}.png')
    
    plt.figure(1)
    plt.plot(epochs, rmse_loss_list_train, label='Training Loss', color='b')
    plt.plot(epochs, rmse_loss_list_test, label='Validation Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'{working_directory}/RMSE_loss_{model_name}_{version}.png')
    
    if save_plots:
        plt.savefig(f'{working_directory}/MSE_loss_{model_name}_{version}.png')

    if save_model:
        torch.save(model.state_dict, f'{working_directory}/{model_name}_{version}.pth')

    ###
    #Performances
    ###
    
    train = VolumeDataset(path_to_csv, Sn=Sn, dataset='train', model_name=model_name)
    train_loader_eval = DataLoader(train, batch_size=104, shuffle=False)
    test = VolumeDataset(path_to_csv, Sn=Sn, dataset='test', model_name=model_name)
    test_loader_eval = DataLoader(test, batch_size=104, shuffle=False)
    
    model.eval()
    loss_list_train = []
    rmse_loss_list_train = []
    with torch.no_grad():
        for batch in train_loader_eval:
            prediction = model(model.prepare_input(batch))
            target = model.prepare_target(batch)
            loss = criterion(prediction, target)
            loss_list_train.append(loss.item())
            rmse_loss = rmse(denorm(prediction), denorm(target))
            rmse_loss_list_train.append(rmse_loss.item())
    loss_list_test = []
    rmse_loss_list_test = []
    with torch.no_grad():
        for batch in test_loader_eval:
            prediction = model(model.prepare_input(batch))
            target = model.prepare_target(batch)
            loss = criterion(prediction, target)
            loss_list_test.append(loss.item())
            rmse_loss = rmse(denorm(prediction), denorm(target))
            rmse_loss_list_test.append(rmse_loss.item())
    
    plt.plot(3)
    plt.plot(range(1, len(loss_list_train) + 1), loss_list_train)
    plt.plot(range(len(loss_list_train) + 1, len(loss_list_train) + len(loss_list_test) + 1), loss_list_test)
    plt.xlabel('Day')
    plt.ylabel('MSE Loss')
    ax = plt.gca()
    ax.set_ylim([0, 2])
    plt.title('Train and test MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    if save_plots:
        plt.savefig(f'{working_directory}/MSE_loss_days_{model_name}_{version}.png')
    
    plt.plot(4)
    plt.plot(range(1, len(rmse_loss_list_train) + 1), rmse_loss_list_train)
    plt.plot(range(len(rmse_loss_list_train) + 1, len(rmse_loss_list_train) + len(rmse_loss_list_test) + 1), rmse_loss_list_test)
    plt.xlabel('Day')
    plt.ylabel('RMSE Loss')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.title('Train and test RMSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    if save_plots:
        plt.savefig(f'{working_directory}/RMSE_loss_days_{model_name}_{version}.png')
    
    print(f'MSE train {sum(loss_list_train)/len(loss_list_train)}, MSE test {sum(loss_list_test)/len(loss_list_test)}')
    print(f'RMSE train {sum(rmse_loss_list_train)/len(rmse_loss_list_train)}, RMSE test {sum(rmse_loss_list_test)/len(rmse_loss_list_test)}')


###
#Models
###

class LSTM_conv_volume_decomp_periodsToEnd(nn.Module):
    def __init__(self, input_size):
        super(LSTM_conv_volume_decomp_periodsToEnd, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10, 1)
        self.conv2d = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(2,5))
        self.conv2 = nn.Conv1d(128, 1, kernel_size=4, stride=2,dilation=1, padding=0)
        self.bn2 = nn.BatchNorm1d(1)

        #
        self.fcp = nn.Linear(input_size, 64)
        self.out = nn.Tanh()  # Output layer (single neuron for regression)

    def forward(self, input):
        x = input
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        #x = self.conv2d(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.out(x)
        return x

    def prepare_input(self, batch):
        #batch, Sn,
        return torch.stack((batch['volume'].to(device), 
                            batch['volume_sd_day_trend'].to(device),
                            batch['volume_sd_day_seasonal'].to(device),
                            batch['volume_sd_week_trend'].to(device),
                            batch['volume_sd_week_seasonal'].to(device),
                            batch['periods_to_end'].to(device)), axis=2)
                           

    def prepare_target(self, batch):
        return batch['total_volume'][:,-1:].to(device)

model_name = 'LSTM_conv_volume_decomp_periodsToEnd'
past_input_n_days = 7
Sn = past_input_n_days*104 + 20
input_size = Sn  # volumes + 1 missing intraday count
model = LSTM_conv_volume_decomp_periodsToEnd(6).to(device)
version = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
batch_size = 100

train_and_eval_model(model=model,
                     input_path=path_to_csv,
                     model_name='LSTM_conv_volume_decomp_periodsToEnd',
                     Sn = Sn,
                     num_epochs=50,
                     optimizer=optimizer,
                     batch_size=batch_size,
                     version=version,
                     save_plots=False,
                     save_model=False)

#A simple fully connected neural network
class FNN(nn.Module):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer (single neuron for regression)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def prepare_input(self, batch):
        missing_intraday = batch['missing_intraday'][:,-1:].to(device)
        volume = batch['volume'].to(device)
        input = torch.cat([volume, missing_intraday], dim=1)
        input = input.to(device)
        return input

    def prepare_target(self, batch):
        #this one might be wrong now
        return batch['total_volume'][:,-1:].to(device)

#FNN
model_name = 'FNN'
past_input_n_days = 7
Sn = past_input_n_days*104 + 1
input_size = Sn + 1  # volumes + 1 missing intraday count
model = FNN(input_size).to(device)
version = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
batch_size = 50

train_and_eval_model(model=model,
                     input_path=path_to_csv,
                     model_name="FNN",
                     Sn = Sn,
                     optimizer=optimizer,
                     batch_size=batch_size,
                     num_epochs=50,
                     version=version,
                     save_plots=False,
                     save_model=False
                     )

#if model_to_run == 'FNN':
#    #A simple fully connected neural network
#    class FNN(nn.Module):
#        def __init__(self, input_size):
#            super(FNN, self).__init__()
#            self.fc1 = nn.Linear(input_size, 64)
#            self.fc2 = nn.Linear(64, 32)
#            self.fc3 = nn.Linear(32, 1)  # Output layer (single neuron for regression)
#
#        def forward(self, x):
#            x = torch.relu(self.fc1(x))
#            x = torch.relu(self.fc2(x))
#            x = self.fc3(x)
#            return x
#
#        def prepare_input(self, batch):
#            missing_intraday = batch['missing_intraday'][:,-1:].to(device)
#            volume = batch['volume'].to(device)
#            input = torch.cat([volume, missing_intraday], dim=1)
#            input = input.to(device)
#            return input
#
#        def prepare_target(self, batch):
#            #this one might be wrong now
#            return batch['total_volume'][:,-1:].to(device)
#
#    #FNN
#    past_input_n_days = 7
#    Sn = past_input_n_days*104 + 1
#    input_size = Sn + 1  # 3*104 volumes + 1 missing intraday count
#    model_name = 'FNN'
#    version = 0
#    model = FNN(input_size).to(device)
#    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#
#elif model_to_run == 'LSTM2':
#class LSTM2(nn.Module):
#    def __init__(self, input_size):
#        super(LSTM2, self).__init__()
#        self.lstm = nn.LSTM(input_size, 63, batch_first=True)
#        self.fc1 = nn.Linear(64, 32)
#        self.fc2 = nn.Linear(32, 16)
#        self.fc3 = nn.Linear(16, 1)  # Output layer (single neuron for regression)
#
#    def forward(self, input):
#        x = input[0]
#        y = input[1]
#        x, _ = self.lstm(x)
#        x = torch.cat([x,y],-1)
#        x = torch.relu(self.fc1(x))
#        x = torch.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
#
#    def prepare_input(self, batch):
#        #batch, Sn, 
#        x = batch['volume'][:,0:-1].squeeze().to(device)
#        y = batch['missing_intraday'][:,-1:].to(device)
#        return (x, y)
#
#    def prepare_target(self, batch):
#        return batch['total_volume'][:,-1:].to(device)
#
##LSTM2
#
#past_input_n_days = 3
#Sn = past_input_n_days*104 + 1 #past record + current record + unknown (to be predicted record)
#input_size = Sn - 1  # 3*104 volumes + 1 missing intraday count
#model_name = 'LSTM2'
#model = LSTM2(input_size).to(device)
#
#elif model_to_run == 'LSTM_multifeatures':
#    class LSTM_multifeatures(nn.Module):
#        def __init__(self, input_size):
#            super(LSTM2, self).__init__()
#            self.lstm = nn.LSTM(input_size, 63, batch_first=True)
#            self.fc1 = nn.Linear(64, 32)
#            self.fc2 = nn.Linear(32, 16)
#            self.fc3 = nn.Linear(16, 1)  # Output layer (single neuron for regression)
#
#        def forward(self, input):
#            x = input[0]
#            y = input[1]
#            x, _ = self.lstm(x)
#            x = torch.cat([x,y],-1)
#            x = torch.relu(self.fc1(x))
#            x = torch.relu(self.fc2(x))
#            x = self.fc3(x)
#            return x
#
#        def prepare_input(self, batch):
#            #batch, Sn, 
#            x = batch['volume'][:,0:-1].squeeze().to(device)
#            y = batch['missing_intraday'][:,-1:].to(device)
#            return (x, y)
#
#        def prepare_target(self, batch):
#            return batch['total_volume'][:,-1:].to(device)
#            
#    #MISSING: Define LSTM multifeature parameters here
#
#elif model_to_run == 'LSTM3':
#    class LSTM3(nn.Module):
#        def __init__(self, input_size):
#            super(LSTM3, self).__init__()
#            self.lstm = nn.LSTM(input_size, 128, batch_first=True)
#            self.fc1 = nn.Linear(15, 64)
#            self.fc2 = nn.Linear(64, 32)
#            self.fc3 = nn.Linear(32, 16)
#            self.fc4 = nn.Linear(16, 1)
#            self.conv1 = nn.Conv1d(1, 1,kernel_size=5,stride=2,dilation=1,padding=0)
#            self.bn1 = nn.BatchNorm1d(1)
#            self.conv2 = nn.Conv1d(1, 1, kernel_size=4, stride=2,dilation=1, padding=0)
#            self.bn2 = nn.BatchNorm1d(1)
#            self.conv3 = nn.Conv1d(1, 1, kernel_size=4, stride=2,dilation=1, padding=0)
#            self.bn3 = nn.BatchNorm1d(1)
#            self.conv4 = nn.Conv1d(1,1,kernel_size=4,stride=3,dilation=1,padding=0)
#            self.out = nn.Tanh()  # Output layer (single neuron for regression)
#
#    def forward(self, input):
#        x = input[0]
#        y = input[1]
#        x, _ = self.lstm(x)
#        x = x.reshape(x.size(0), 1, x.size(1))
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.conv2(x)
#        x = self.bn2(x)
#        x = self.conv3(x)
#        x = self.bn3(x)
#        x = x.reshape(x.size(0), x.size(2))
#        x =torch.cat([x,y],-1)
#
#        x = torch.relu(self.fc1(x))
#        x = torch.relu(self.fc2(x))
#        x = torch.relu(self.fc3(x))
#        x = self.fc4(x)
#        x = self.out(x)
#        return x
#
#    def prepare_input(self, batch):
#        #batch, Sn,
#        x = batch['volume'][:,0:-1].squeeze().to(device)
#        y = batch['missing_intraday'][:,-1:].to(device)
#        return (x, y)
#
#    def prepare_target(self, batch):
#        return batch['total_volume'][:,-1:].to(device)
#
#    #LSTM3
#    version = '0'
#    batch_size = 50
#    past_input_n_days = 3
#    Sn = past_input_n_days*104 + 2 #past record + current record + unknown (to be predicted record)
#    input_size =  Sn - 1 # 3*104 volumes + 1 missing intraday count
#    model_name = 'LSTM3'
#    model = LSTM3(input_size).to(device)
#    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#
#elif model_to_run == 'LSTM4input': 
#
#    class LSTM4input(nn.Module):
#        def __init__(self, input_size):
#            super(LSTM4input, self).__init__()
#            self.lstm = nn.LSTM(input_size, 32, batch_first=True)
#            self.fc1 = nn.Linear(32, 16)
#            self.fc2 = nn.Linear(16, 8)
#            self.fc3 = nn.Linear(8, 1)  # Output layer (single neuron for regression)
#
#        def forward(self, input):
#            x = input
#            x, _ = self.lstm(x)
#            x = torch.relu(self.fc1(x[:,-1,:]))
#            x = torch.relu(self.fc2(x))
#            x = self.fc3(x)
#            return x
#
#        def prepare_input(self, batch):
#            #batch, Sn, 
#            return torch.stack((batch['volume'].to(device), batch['periods_to_end'].to(device), batch['week_day'].to(device), batch['month'].to(device)), axis=2)
#
#        def prepare_target(self, batch):
#            return batch['total_volume'][:,-1:].to(device)
#
#    #LSTM4input
#    version = '0'
#    batch_size = 50
#    past_input_n_days = 3
#    Sn = past_input_n_days*104 + 1 #past record + current record + unknown (to be predicted record)
#    input_size = 4  # 3*104 volumes + 1 missing intraday count
#    model_name = 'LSTM4input'
#    model = LSTM4input(input_size).to(device)
#    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#
#if model_to_run == 'EXPERIMENT':
#    
#    #This is just a place-holder used to show how to prepare the input data
#    class EXPERIMENT(nn.Module):
#        def __init__(self, input_size):
#            super(EXPERIMENT, self).__init__()
#            self.lstm = nn.LSTM(input_size, 64, batch_first=True)
#            self.fc1 = nn.Linear(64, 32)
#            self.fc2 = nn.Linear(32, 16)
#            self.fc3 = nn.Linear(16, 1)  # Output layer (single neuron for regression)
#    
#        def forward(self, input):
#            x = input
#            x, _ = self.lstm(x)
#            x = torch.relu(self.fc1(x[:,-1,:]))
#            x = torch.relu(self.fc2(x))
#            x = self.fc3(x)
#            return x
#    
#        def prepare_input(self, batch):
#            #batch, Sn, 
#            return torch.cat((batch['volume'].unsqueeze(-1), 
#                              batch['periods_to_end'].unsqueeze(-1), 
#                              batch['dw_one_hot'], 
#                              batch['m_one_hot']), 
#                              axis=2).to(device)
#            #return torch.cat((batch['volume'].unsqueeze(-1), 
#            #                  batch['periods_to_end'].unsqueeze(-1)), 
#            #                  axis=2).to(device)
#    
#        def prepare_target(self, batch):
#            return batch['total_volume'][:,-1:].to(device)
#    
#    #EXPERIMENT
#    version = '0'
#    batch_size = 20
#    past_input_n_days = 10
#    Sn = past_input_n_days*104 + 1 #past record + current record + unknown (to be predicted record)
#    input_size = 19
#    model_name = 'EXPERIMENT'
#    model = EXPERIMENT(input_size).to(device)
#    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#
#
#class RMSE(nn.Module):
#    def forward(self, y_pred, y_true):
#        squared_error = (y_pred - y_true) ** 2
#        ratio = squared_error/y_true**2
#        return torch.sqrt(ratio.mean())
#
#criterion = nn.MSELoss()
#rmse = RMSE()
##optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
#train = VolumeDataset(path_to_csv, Sn=Sn, dataset='train', model_name=model_name)
#train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
#test = VolumeDataset(path_to_csv, Sn=Sn, dataset='validation', model_name=model_name)
#test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
#denorm = train.total_volume_normalizer.denormalize
#
####
##TRAINING
####
#
#def train(data_loader, model, loss_fn, optimizer):
#    model.train()
#    loss_list = []
#    rmse_loss_list = []
#    #import time
#    #start = time.time()
#    for batch in data_loader:
#        prediction = model(model.prepare_input(batch))
#        target = model.prepare_target(batch)
#        loss = loss_fn(prediction, target)
#        loss_list.append(loss.item())
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        with torch.no_grad():
#            rmse_loss = rmse(denorm(prediction), denorm(target))
#            rmse_loss_list.append(rmse_loss.item())
#    average_loss = sum(loss_list)/len(loss_list)
#    average_rmse_loss = sum(rmse_loss_list)/len(rmse_loss_list)
#    return average_loss, average_rmse_loss
#
#def test(data_loader, model, loss_fn):
#    model.eval()
#    loss_list = []
#    rmse_loss_list = []
#    with torch.no_grad():
#        for batch in data_loader:
#            prediction = model(model.prepare_input(batch))
#            target = model.prepare_target(batch)
#            loss = loss_fn(prediction, target)
#            loss_list.append(loss.item())
#            rmse_loss = rmse(denorm(prediction), denorm(target))
#            rmse_loss_list.append(rmse_loss.item())
#        average_loss = sum(loss_list)/len(loss_list)
#        average_rmse_loss = sum(rmse_loss_list)/len(rmse_loss_list)
#        return average_loss, average_rmse_loss
#
#loss_list_train = []
#rmse_loss_list_train = []
#loss_list_test = []
#rmse_loss_list_test = []
#for epoch in range(num_epochs):
#    average_loss_train, average_rmse_loss_train = train(train_loader, model, criterion, optimizer)
#    average_loss_test, average_rmse_loss_test = test(test_loader, model, criterion)
#    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_loss_train}, Validation Loss: {average_loss_test}, RMSE Train Loss: {average_rmse_loss_train}, RMSE Validation Loss: {average_rmse_loss_test}")
#    loss_list_train.append(average_loss_train)
#    rmse_loss_list_train.append(average_rmse_loss_train)
#    loss_list_test.append(average_loss_test)
#    rmse_loss_list_test.append(average_rmse_loss_test)
#
##do plot
#epochs = range(1, num_epochs + 1)
#plt.figure(0)
#plt.plot(epochs, loss_list_train, label='Training Loss', color='b')
#plt.plot(epochs, loss_list_test, label='Validation Loss', color='r')
#plt.xlabel('Epochs')
#plt.ylabel('MSE Loss')
#plt.title('Training Loss vs. Epochs')
#plt.legend()
#plt.grid(True)
#plt.show()
#plt.savefig(f'MSE_loss_{model_name}_{version}.png')
#
#plt.figure(1)
#plt.plot(epochs, rmse_loss_list_train, label='Training Loss', color='b')
#plt.plot(epochs, rmse_loss_list_test, label='Validation Loss', color='r')
#plt.xlabel('Epochs')
#plt.ylabel('RMSE Loss')
#plt.title('Training Loss vs. Epochs')
#plt.legend()
#plt.grid(True)
#plt.show()
#plt.savefig(f'RMSE_loss_{model_name}_{version}.png')
#
#working_directory = os.path.dirname(os.path.realpath(__file__))
#torch.save(model.state_dict, f'{working_directory}/{model_name}_{version}.pth')
####
##Performances
####
#
#train = VolumeDataset(path_to_csv, Sn=Sn, dataset='train', model_name=model_name)
#train_loader_eval = DataLoader(train, batch_size=104, shuffle=False)
#test = VolumeDataset(path_to_csv, Sn=Sn, dataset='test', model_name=model_name)
#test_loader_eval = DataLoader(test, batch_size=104, shuffle=False)
#
#model.eval()
#loss_list_train = []
#rmse_loss_list_train = []
#with torch.no_grad():
#    for batch in train_loader_eval:
#        prediction = model(model.prepare_input(batch))
#        target = model.prepare_target(batch)
#        loss = criterion(prediction, target)
#        loss_list_train.append(loss.item())
#        rmse_loss = rmse(denorm(prediction), denorm(target))
#        rmse_loss_list_train.append(rmse_loss.item())
#loss_list_test = []
#rmse_loss_list_test = []
#with torch.no_grad():
#    for batch in test_loader_eval:
#        prediction = model(model.prepare_input(batch))
#        target = model.prepare_target(batch)
#        loss = criterion(prediction, target)
#        loss_list_test.append(loss.item())
#        rmse_loss = rmse(denorm(prediction), denorm(target))
#        rmse_loss_list_test.append(rmse_loss.item())
#
#plt.plot(3)
#plt.plot(range(1, len(loss_list_train) + 1), loss_list_train)
#plt.plot(range(len(loss_list_train) + 1, len(loss_list_train) + len(loss_list_test) + 1), loss_list_test)
#plt.xlabel('Day')
#plt.ylabel('MSE Loss')
#ax = plt.gca()
#ax.set_ylim([0, 2])
#plt.title('Train and test MSE Loss')
#plt.legend()
#plt.grid(True)
#plt.show()
#plt.savefig(f'MSE_loss_days_{model_name}_{version}.png')
#
#plt.plot(4)
#plt.plot(range(1, len(rmse_loss_list_train) + 1), rmse_loss_list_train)
#plt.plot(range(len(rmse_loss_list_train) + 1, len(rmse_loss_list_train) + len(rmse_loss_list_test) + 1), rmse_loss_list_test)
#plt.xlabel('Day')
#plt.ylabel('RMSE Loss')
#ax = plt.gca()
#ax.set_ylim([0, 1])
#plt.title('Train and test RMSE Loss')
#plt.legend()
#plt.grid(True)
#plt.show()
#plt.savefig(f'RMSE_loss_days_{model_name}_{version}.png')
#
#print(f'MSE train {sum(loss_list_train)/len(loss_list_train)}, MSE test {sum(loss_list_test)/len(loss_list_test)}')
#print(f'RMSE train {sum(rmse_loss_list_train)/len(rmse_loss_list_train)}, RMSE test {sum(rmse_loss_list_test)/len(rmse_loss_list_test)}')
#
#
##average_loss = sum(loss_list)/len(loss_list)
##return average_loss
#
###EXPERIMENT
##version = '0'
##batch_size = 20
##past_input_n_days = 10
##Sn = past_input_n_days*104 + 1 #past record + current record + unknown (to be predicted record)
##input_size = 19
##model_name = 'EXPERIMENT'
##model = EXPERIMENT(input_size).to(device)
##optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#
#