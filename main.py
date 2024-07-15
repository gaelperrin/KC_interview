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



device = "cuda" if torch.cuda.is_available() else "cpu"

class VolumeDataset(Dataset):
    def __init__(self, path_to_csv, Sn, train=True, last_training_date="2022-03-09"):

        self.data = pd.read_csv('dataKCx.csv')
        self.Sn = Sn
        self.train = train

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

        #separated data in test and train
        self.train_dataset = self.data[self.data['date'] <= last_training_date]
        self.test_dataset = self.data[self.data['date'] > last_training_date]

        #get the target
        self.total_volume_per_day = self.data.groupby('date')['volume'].sum()

        ####
        ##How to do it ? Problem with outliers

        ###normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_dataset.loc[:, 'volume'] = self.scaler.fit_transform(self.train_dataset[['volume']])
        self.test_dataset.loc[:, 'volume'] = self.scaler.transform(self.test_dataset[['volume']])

        #self.scaler.inverse_transform(self.total_volume_per_day_norm)[:, [0]]

    def __len__(self):
        if self.train:
            return len(self.train_dataset) - self.Sn + 1
        else:
            return len(self.test_dataset) - self.Sn + 1

    def __getitem__(self, idx):
        if self.train:
            #all past Sn volumes
            output_dataset = self.train_dataset[idx:idx + self.Sn]
        else:
            output_dataset = self.test_dataset[idx:idx + self.Sn]
        #period number k and date number d (as in instructions) used in model performance evaluation
        period_k = output_dataset['k'].iloc[-1]
        date_d = output_dataset['d'].iloc[-1]
        volume = output_dataset['volume']
        #_last_date = output_dataset['date'].iloc[-1]
        #note: missing intraday also include opening and closing
        #missing_intraday = 104 - len(output_dataset[output_dataset['date'] == _last_date])
        #total_volume = self.total_volume_per_day.loc[_last_date]
        return {
            'volume' : torch.tensor(volume.values, dtype=torch.float32), 
        }

###
#Models
###

class LSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=10, num_layers=1, batch_first=True)
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def prepare_input(self, batch):
        return torch.narrow(batch['volume'], 1, 0, batch['volume'].shape[1] - 1).to(device)

    def prepare_target(self, batch):
        return torch.narrow(batch['volume'], 1, batch['volume'].shape[1] - 1, 1).to(device)

class TotalVolumeEstimator(nn.Module):
    def __init__(self, input_size):
        super(TotalVolumeEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer (single neuron for regression)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def prepare_input(self, batch):
        missing_intraday = batch['missing_intraday'].to(device)
        volume = batch['volume'].to(device)
        input = torch.cat([volume, missing_intraday.unsqueeze(-1)], dim=1)
        input = input.to(device)
        return input
    
    def prepare_target(self, batch):
        return batch['total_volume'].to(device).unsqueeze(-1)



batch_size = 100
past_input_n_days = 3
#Sn = past_input_n_days*104 + 1
#input_size = Sn + 1  # 3*104 volumes + 1 missing intraday count
#model = TotalVolumeEstimator(input_size).to(device)
Sn = past_input_n_days*104 + 2 #past record + current record + unknown (to be predicted record)
#Sn  =  2
input_size = Sn - 1  # 3*104 volumes + 1 missing intraday count
model = LSTM(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

train = VolumeDataset('dataKCx.csv', Sn=Sn, train=True, last_training_date="2021-5-08")
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test = VolumeDataset('dataKCx.csv', Sn=Sn, train=False, last_training_date="2021-5-08")
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

num_epochs=100

###
#TRAINING
###

def train(data_loader, model, loss_fn, optimizer):
    model.train()
    loss_list = []
    import time
    start = time.time()
    for batch in data_loader:
        prediction = model(model.prepare_input(batch))
        target = model.prepare_target(batch)
        loss = loss_fn(prediction, target)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Took {time.time() - start}')
        start = time.time()
    average_loss = sum(loss_list)/len(loss_list)
    return average_loss

def test(data_loader, model, loss_fn):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in data_loader:
            prediction = model(model.prepare_input(batch))
            target = model.prepare_target(batch)
            loss = loss_fn(prediction, target)
            #predicted_total_volume = model(model.prepare_input(batch))
            #total_volume = batch['total_volume'].to(device)
            #loss = loss_fn(predicted_total_volume, total_volume.unsqueeze(-1))
            loss_list.append(loss.item())
        average_loss = sum(loss_list)/len(loss_list)
        return average_loss

for epoch in range(num_epochs):
    average_loss_train = train(train_loader, model, criterion, optimizer)
    average_loss_test = test(test_loader, model, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_loss_train}, Test Loss: {average_loss_test}")
print('pass')

###
#Performances
###

train = VolumeDataset('dataKCx.csv', Sn=Sn, train=True, last_training_date="2021-5-08")
train_loader_eval = DataLoader(train, batch_size=104, shuffle=False)
test = VolumeDataset('dataKCx.csv', Sn=Sn, train=False, last_training_date="2021-5-08")
test_loader_eval = DataLoader(test, batch_size=104, shuffle=False)

model.eval()
loss_list = []
#counter = 1
loss_list_train = []
for (volume, missing_intraday, total_volume, period_k, date_d) in train_loader_eval:
    volume = volume.to(device)
    missing_intraday = missing_intraday.to(device)
    total_volume = total_volume.to(device)
    input = torch.cat([volume, missing_intraday.unsqueeze(-1)], dim=1)
    input = input.to(device)
    predicted_total_volume = model(input)
    predicted_total_volume_ratio = torch.div(predicted_total_volume.squeeze(), total_volume)
    loss = torch.sqrt(criterion(predicted_total_volume_ratio, torch.ones(104).to(device)))
    loss_list_train.append(loss.item())

loss_list_test = []
for (volume, missing_intraday, total_volume, period_k, date_d) in test_loader_eval:
    volume = volume.to(device)
    missing_intraday = missing_intraday.to(device)
    total_volume = total_volume.to(device)
    input = torch.cat([volume, missing_intraday.unsqueeze(-1)], dim=1)
    input = input.to(device)
    predicted_total_volume = model(input)
    predicted_total_volume_ratio = torch.div(predicted_total_volume.squeeze(), total_volume)
    loss = torch.sqrt(criterion(predicted_total_volume_ratio, torch.ones(104).to(device)))
    loss_list_test.append(loss.item())

#save model
#import os
#working_directory = os.path.dirname(os.path.realpath(__file__))
#torch.save(model.state_dict, f'{working_directory}/FFN.pth')

plt.scatter(range(1, len(loss_list_train) + 1), loss_list_train)
plt.scatter(range(len(loss_list_train) + 1, len(loss_list_train) + len(loss_list_test) + 1), loss_list_test)
print(f'RMSE train {sum(loss_list_train)/len(loss_list_train)}, RMSE test {sum(loss_list_test)/len(loss_list_test)}')


#average_loss = sum(loss_list)/len(loss_list)
#return average_loss

