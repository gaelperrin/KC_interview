import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

path_to_csv = 'dataKCx.csv'

raw_pdf = pd.read_csv(path_to_csv)

raw_pdf.rename(columns = {"Unnamed: 0":"id"}, inplace=True)

##Add dummy timestamp
opening_time = pd.Timestamp('06:55:00').strftime("%H:%M:%S")
raw_pdf.loc[raw_pdf['tradingphase'] == 'OPENING AUCTION','time'] = opening_time
closing_time = pd.Timestamp('15:30:00').strftime("%H:%M:%S")
raw_pdf.loc[raw_pdf['tradingphase'] == 'CLOSING AUCTION','time'] = closing_time

#period k and date d integers used in final performance evaluation
raw_pdf['k'] = (raw_pdf['id']-1)%104 + 1
raw_pdf['d'] = (raw_pdf['id'] -1)//104 + 1

c_pdf = raw_pdf[(raw_pdf['tradingphase'] == 'CONTINUOUS') | (raw_pdf['tradingphase'] == 'OPENING AUCTION')]

total_volume_per_day = raw_pdf.groupby('date')['volume'].sum()
raw_pdf = raw_pdf.merge(total_volume_per_day, on='date', how='left')
raw_pdf.rename(columns = {"volume_x":"volume", "volume_y":"total_volume"}, inplace=True)
raw_pdf['total_volume_yesterday'] = raw_pdf['total_volume'].shift(104)
raw_pdf['total_volume_diff'] = raw_pdf['total_volume'] - raw_pdf['total_volume_yesterday']

#get data daily data for simple models
total_volume_per_day = raw_pdf[['date', 'total_volume']].merge(raw_pdf[['date', 'd', 'total_volume']], on='date', how='left')

raw_pdf['date_time'] = pd.to_datetime(raw_pdf['date'] + 'T' + raw_pdf['time'])
raw_pdf['m'] = raw_pdf['date_time'].dt.month
raw_pdf['dw'] = raw_pdf['date_time'].dt.dayofweek

raw_pdf['volume_yesterday'] = raw_pdf['volume'].shift(104)
raw_pdf['volume_diff'] = raw_pdf['volume'] - raw_pdf['volume_yesterday']

raw_pdf = raw_pdf.set_index('id')

train_data = raw_pdf[raw_pdf['date'] <= "2021-5-08"]
test_data = raw_pdf[raw_pdf['date'] > "2021-5-08"]
train_data_daily = train_data.drop_duplicates(['d']).reset_index()
test_data_daily = test_data.drop_duplicates(['d']).reset_index()

#performence if just using mean
train_data_daily['mean_volume'] = train_data_daily['total_volume'].mean()
np.sqrt(((train_data_daily['mean_volume'] - train_data_daily['total_volume'])/train_data_daily['total_volume'])**2).mean()

volume_aggreation = train_data.groupby('k')['volume'].mean()
volume_aggreation_month = train_data.groupby('m')['volume'].mean()
volume_aggreation_dayofweek = train_data.groupby('dw')['volume'].mean()


def do_plot(dataset):
    plt.plot(dataset.index, dataset)
    plt.show()

def get_daily_data(dataset, index, train_size, validation_size):
    window_size = train_size + validation_size 
    n = len(dataset)
    validation_data = dataset[index : index + window_size]
    train_data = validation_data[: - validation_size]
    remaining_samples = n - index - window_size
    return train_data, validation_data, remaining_samples

index = 0
#uncommment to try other parameters
model_parameters_list = [
#    [(1, 2, 1), (1, 2, 1, 6)], #average MSE: 0.3746707416155054 
#    [(1, 2, 1), (1, 1, 1, 6)], #average MSE: 0.286140910414454
#    [(1, 2, 1), (1, 0, 1, 6)], #average MSE: 0.45770890271557335 
    [(1, 1, 1), (1, 1, 1, 6)],  #average MSE: 0.2665712373963622 <-- best
#    [(1, 0, 1), (1, 1, 1, 6)],  #average MSE: 1.142139652555844
#    [(1, 1, 1), (2, 1, 1, 6)],  #average MSE: 0.26850059809350446
#    [(1, 1, 1), (3, 1, 1, 6)],  #average MSE 0.9213187228138766
] 
model_parameters_list = []
volume_column = 'total_volume'
validation_length = 1
make_plots_at_index = 50
for model_parameters in model_parameters_list:
    mse_list = []
    remaining = 1
    while remaining > 0:
        train, validation, remaining = get_daily_data(train_data_daily, index, 3*21, validation_length)
        #do fit 
        model = SARIMAX(train[volume_column],  order = model_parameters[0],  seasonal_order = model_parameters[1])
        fitted_model = model.fit(disp=False)
        #compute MSE
        validation_target = validation[-validation_length:]
        forcast = fitted_model.forecast(validation_length)
        MSE = np.sqrt(((((validation_target[volume_column] - forcast)/validation_target[volume_column])**2).mean()))
        #print(f'MSE: {MSE}')
        mse_list.append(MSE)
        #plot t+1 prediction
        #just to get visuals in some random steps
        if index == make_plots_at_index:
            plt.figure(0)
            result = fitted_model.fittedvalues
            result.plot(legend=True)
            plt.plot(train.index, train[volume_column])
            plot_acf(train[volume_column].dropna(), lags=20)
            plot_pacf(train[volume_column].dropna(), lags=20)
        index += 1
        if remaining == 0:
            break
    print(f'Model parameters: {model_parameters}, average MSE: {sum(mse_list)/len(mse_list)}')

def do_ARIMAX_loop(volume_column, validation_length, make_plots_at_index, model_parameters, dataset, train_window_size=21*3, optimizing_parameters=False):
    #volume_column = 'total_volume'
    #validation_length = 1
    #make_plots_at_index = 50
    #if optimizing parameters is true, no inference to be done but justcheck MSE
    #list of predictions from the model
    predictions_list = []
    for model_parameters in model_parameters_list:
        mse_list = []
        remaining = 1
        index = 0
        while remaining > 0:
            #get data to "train" SARIMAX, validation to compare SARIMAX prediction and "remaining" variable to stop at last batch
            train, validation, remaining = get_daily_data(dataset, index, train_window_size, validation_length)
            #do fit 
            model = SARIMAX(train[volume_column],  order = model_parameters[0],  seasonal_order = model_parameters[1])
            fitted_model = model.fit(disp=False)
            #compute MSE
            validation_target = validation[-validation_length:]
            forcast = fitted_model.forecast(validation_length)
            MSE = np.sqrt(((((validation_target[volume_column] - forcast)/validation_target[volume_column])**2).mean()))
            #print(f'MSE: {MSE}')
            mse_list.append(MSE)
            #plot t+1 prediction
            #just to get visuals in some random steps
            if index == make_plots_at_index and optimizing_parameters:
                plt.figure(0)
                result = fitted_model.fittedvalues
                result.plot(legend=True)
                plt.plot(train.index, train[volume_column])
                plot_acf(train[volume_column].dropna(), lags=20)
                plot_pacf(train[volume_column].dropna(), lags=20)
                plt.show()

            if not optimizing_parameters:
                predictions_list.append(forcast.values[0])
            index += 1
            if remaining == 0:
                break
        print(f'Model parameters: {model_parameters}, average MSE: {sum(mse_list)/len(mse_list)}, percentile of MSE: {np.percentile(mse_list, q=95)}')

        if not optimizing_parameters:
            predictions_list_temp = [0]*(len(dataset) - (len(predictions_list))) + predictions_list[:]
            dataset['total_volume_predicted'] = pd.Series(predictions_list_temp)
            return dataset


model_parameters = [(1, 1, 1), (1, 1, 1, 6)]
volume_column = 'total_volume'
validation_length = 1
make_plots_at_index = 50
#optimize ARIMAX parameters by doing a scan, uncomment list to try more parameters
model_parameters_list = [
#    [(1, 2, 1), (1, 2, 1, 6)], #average MSE: 0.3746707416155054 
#    [(1, 2, 1), (1, 1, 1, 6)], #average MSE: 0.286140910414454
#    [(1, 2, 1), (1, 0, 1, 6)], #average MSE: 0.45770890271557335 
#    [(1, 1, 1), (1, 1, 1, 6)],  #average MSE: 0.2665712373963622 <-- best
#    [(1, 0, 1), (1, 1, 1, 6)],  #average MSE: 1.142139652555844
#    [(1, 1, 1), (2, 1, 1, 6)],  #average MSE: 0.26850059809350446
#    [(1, 1, 1), (3, 1, 1, 6)],  #average MSE 0.9213187228138766
#    [(2, 1, 1), (3, 1, 1, 6)],  #average MSE 0.2668491210520514, percentile of MSE: 0.6864501866415369
#    [(2, 1, 1), (1, 1, 1, 6)],  #average MSE: 0.2668491210520514, percentile of MSE: 0.686450186641536
    [(1, 1, 1), (1, 1, 1, 6)],  #average MSE: 0.2668491210520514, percentile of MSE: 0.686450186641536
] 
train = do_ARIMAX_loop(volume_column, validation_length, make_plots_at_index=50, model_parameters=model_parameters_list, dataset=train_data_daily, train_window_size=21*3, optimizing_parameters=True)
#apply ARIMAX to training and testing sample
train = do_ARIMAX_loop(volume_column, validation_length, make_plots_at_index=-1, model_parameters=model_parameters, dataset=train_data_daily, train_window_size=21*3)
plt.figure(0)
train['total_volume'].plot()
train['total_volume_predicted'].plot()
test = do_ARIMAX_loop(volume_column, validation_length, make_plots_at_index=-1, model_parameters=model_parameters, dataset=test_data_daily, train_window_size=21*3)
plt.figure(0)
train['test'].plot()
train['test'].plot()


total_volume_train = train_data
do_plot(train_data_daily['total_volume'][0:3*28])

#do_plot(train_data['total_volume'])
model = SARIMAX(train_data_daily['total_volume'][0:3*28],  order = (1, 2, 1),  seasonal_order =(1, 1, 1, 6))
fitted_model = model.fit()
plt.figure(0)
result = fitted_model.fittedvalues
result.plot(legend=True)
plt.plot(train_data_daily['total_volume'][0:3*28].index, train_data_daily['total_volume'][0:3*28])
fitted_model.forecast(1)
plt.figure(1)
train_prediction = fitted_model.predict(start=0, end=len(train_data_daily['total_volume'][0:3*28]) + 5)
train_prediction.plot(legend=True)
plt.plot(range(len(data.index)), data['volume_diff'])





class LoadData():
    def __init__(self, dataset, n_days_continous, continous_value='volume_diff', ):
        self.dataset = dataset
        self.dataset_continous = self.dataset[(self.dataset['k'] != 1) & (self.dataset['k'] != 80) & (self.dataset['k'] != 104)]
        self.dataset_continous = self.dataset_continous.reset_index()
        self.n_days_continous = n_days_continous
        self.continous_value = continous_value

    def get_dailyvolume_data(self, latest_index):
        """
        latest_index: index corresponding to the last volume entry we have and from which we make the prediction (k in the pdf)
        """
        #Information about latest index
        latest_index_info = self.dataset.iloc[latest_index]
        #Transform index to fetch n_days_continuous of past continuous volumes without 1, 80 and 104 
        end_day_index = ((latest_index)//104 + 1)*104 - 1
        daily_volume_end_index = end_day_index- (end_day_index//104)*3 - (end_day_index%104 + 1)//80 -1
        daily_volume_index = latest_index - (latest_index//104)*3 - (latest_index%104 + 1)//80 - 1
        #Get past n_days of continuous volumes + volumes until the end of the periode (for validation)
        daily_volume_dataset = self.dataset_continous.iloc[daily_volume_index - self.n_days_continous*101: daily_volume_end_index]
        #daily_volume_dataset = daily_volume_dataset[self.continous_value]
        daily_volume_dataset = daily_volume_dataset
        daily_volume_dataset_train = daily_volume_dataset[:self.n_days_continous*101 + 1]
        #How many steps of continous volumes (without 1, 80 and 104) we need to predict
        remaining_daily = len(daily_volume_dataset) - len(daily_volume_dataset_train)
        return daily_volume_dataset_train, daily_volume_dataset, remaining_daily, latest_index_info

continous_value = 'volume_diff'
#how long do we look back to predict the continous volumes
n_days_continous = 10
#make sure enough past data for predictions
n_start_batch = n_days_continous + 1
loaded_data = LoadData(train_data, n_days_continous=n_days_continous)

#variable used in loop
fitted_model_continous = None
latest_value = None
previous_prediction = None

for batch in range(n_start_batch, n_start_batch+10):
    for k in range(0, 104):
        index = batch*104 + k
        #get data to predict continous volumes (without 1, 80 and 104)
        train, data, n_to_predict, index_info = loaded_data.get_dailyvolume_data(index)
        print(f'batch: {batch}, k: {k}')
        print(index_info['k'], index_info['date_time'])
        print('test')
        #refit (if necessary) and predict
        if fitted_model_continous == None or index//(104*3) == 0:
            model = SARIMAX(train[continous_value],  order = (1, 1, 1),  seasonal_order =(1, 1, 1, 101))
            fitted_model_continous = model.fit()
        else:
            #in case no update in continous transaction, no need to predict again
            if train[-1:] == latest_value:
                fitted_model_continous = previous_prediction
            else:
                fitted_model_continous = fitted_model_continous.append(train[-1:])
                fitted_model_continous = fitted_model_continous.remove_data(train[0:1])
        latest_value = train[-1:]
        forcast = fitted_model_continous.forecast(n_to_predict)

fitted_model = None
for latest_index in range(4000, 10000):
    train, data, n_to_predict = loaded_data.get_data(latest_index)
    if fitted_model == None or latest_index//(3*104) == 0:
        model = SARIMAX(train,  order = (1, 1, 1),  seasonal_order =(1, 1, 1, 101))
        fitted_model = model.fit()
        forcast = fitted_model.forecast(n_to_predict)
    else:
        fitted_model = fitted_model.append(train[-1:])
    forcast = fitted_model.forecast(n_to_predict)
    validation = data.iloc[-n_to_predict:]
    rmse = ((forcast - validation)**2).mean()**0.5
    print(rmse)




train, data, n_to_predict = loaded_data.get_data(3040)

def do_plot(dataset):
    plt.plot(dataset.index, dataset)
    plt.show()

#do_plot(data['volume_diff'])

model = SARIMAX(train['volume_diff'],  order = (1, 1, 1),  seasonal_order =(1, 1, 1, 101))
fitted_model = model.fit()

forcast = fitted_model.forecast(n_to_predict)
validation = data.iloc[-n_to_predict:]
rmse = ((forcast - validation['volume_diff'])**2).mean()**0.5


#plt.figure(0)
#result = fitted_model.fittedvalues
#result.plot(legend=True):wa
#plt.plot(data.index, data['volume_diff'])

plt.figure(0)
result = fitted_model.fittedvalues
result.plot(legend=True)
plt.plot(data.index, data['volume_diff'])

plt.figure(1)
train_prediction = fitted_model.predict(start=0, end=len(train) + n_to_predict - 1)
train_prediction.plot(legend=True)
plt.plot(range(len(data.index)), data['volume_diff'])

plt.figure(3)
train_prediction = fitted_model.predict(start=len(train), end=len(train) + n_to_predict)
train_prediction.plot(legend=True)
plt.plot(range(len(train), 1, len(train) +n_to_predict), data['volume_diff'].iloc[:n_to_predict])


