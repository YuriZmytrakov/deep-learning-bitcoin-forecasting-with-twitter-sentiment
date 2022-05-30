# Imports
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from numpy import hstack

print("################## Load data ##################")
cwd = os.getcwd()
formatted_bitcoin_tweets_file = os.getcwd() + "\\Data\\formatted_bitcoin_tweets_labelled.csv"
formatted_bitcoin_price_file = os.getcwd() + "\\Data\\formatted_bitcoin_price.csv"
df_price = pd.read_csv(formatted_bitcoin_price_file)
df_price = df_price.set_index(pd.DatetimeIndex(df_price['date'])) #Moving time data into index
df_price.drop(['date'], inplace=True, axis=1)

df_tweets = pd.read_csv(formatted_bitcoin_tweets_file)
df_tweets = df_tweets.set_index(pd.DatetimeIndex(df_tweets['date'])) #Moving time data into index
df_tweets = df_tweets[['sentiment']]

df_tweets['sentiment'].groupby(pd.Grouper(freq="M")).value_counts().unstack().plot(kind="bar")

print("################## Rollup sentiment by day ##################")

def numpy_scaled(series):
    num_0 = np.sum(series.values == 0)
    num_1 = np.sum(series.values == 1)
    max_val = max(num_0, num_1)
    if max_val == 0:    # So we don't divide by 0
        return 0
    return (num_1-num_0) / max_val

df_tweets = df_tweets.resample('D').apply(numpy_scaled)

print("################## Merge frames ##################")

df_price = df_price.merge(df_tweets, on='date', how='left')
df_price['close'] = np.roll(df_price['close'], 1)

print("################## Remove extra fields ##################")
df_price.drop('target', axis=1, inplace=True)
df_price.drop('open', axis=1, inplace=True)

print("################## Load definitions ##################")

def Calc_MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

import math
def build_LSTM_sentiment(lookback_window, neurons, opt):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(lookback_window, 2)))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

def build_LSTM_stacked_sentiment(lookback_window, neurons, opt):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', return_sequences = True, input_shape=(lookback_window, 2)))
    model.add(LSTM(math.floor(neurons/2)))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

def build_LSTM_bidirectional_sentiment(lookback_window, neurons, opt): 
    model = Sequential()
    model.add(Bidirectional(LSTM(neurons, activation='relu', return_sequences = True, input_shape=(lookback_window, 2))))
    model.add(Bidirectional(LSTM(math.floor(neurons/2))))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

def build_GRU_sentiment(lookback_window, neurons, opt):
    model = Sequential()
    model.add(GRU(neurons, activation='relu', input_shape=(lookback_window, 2)))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

def build_GRU_stacked_sentiment(lookback_window, neurons, opt):
    model = Sequential()
    model.add(GRU(neurons, activation='relu', return_sequences = True, input_shape=(lookback_window, 2)))
    model.add(GRU(math.floor(neurons/2)))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

def build_GRU_bidirectional_sentiment(lookback_window, neurons, opt):
    model = Sequential()
    model.add(Bidirectional(GRU(neurons, activation='relu', return_sequences = True, input_shape=(lookback_window, 2))))
    model.add(Bidirectional(GRU(math.floor(neurons/2))))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model


# (40, 'adam', 50, 90, 25)
# Grab best hyperparameter combination from tuning run
neurons = 40
opt = 'adam'
epochs = 50
prediction_horizon = 90
lookback_window = 25

print("################## Run against models ##################")

# LSTM input is of the form (n_samples, lookback_window, n_features)
X_train_price = df_price['close'].values[:-prediction_horizon]
X_test_price = df_price['close'].values[-(lookback_window + prediction_horizon):]
y_test = df_price['close'].values[-prediction_horizon:]
X_train_sentiment = df_price['sentiment'].values[:-prediction_horizon].reshape(-1, 1)
X_test_sentiment = df_price['sentiment'].values[-(lookback_window + prediction_horizon):].reshape(-1, 1)

naive_prediction = df_price["close"].values[-(prediction_horizon+prediction_horizon):-prediction_horizon]

scaler = StandardScaler()
scaler.fit(X_train_price.reshape(-1, 1))
X_train_close_price = scaler.transform(X_train_price.reshape(-1, 1))
X_test_close_price = scaler.transform(X_test_price.reshape(-1, 1))
y_test = y_test.reshape(-1, 1)

X_train = hstack((X_train_close_price, X_train_sentiment))
X_test = hstack((X_test_close_price, X_test_sentiment))
Y_train = X_train_close_price

# Convert  the data from sequence to supervised data - [2, 3, 4, 5, 4, 6, 7] = [2, 3, 4] -> [5]
train_generator = TimeseriesGenerator(X_train, Y_train, length=lookback_window, batch_size=prediction_horizon)
test_generator = TimeseriesGenerator(X_test, X_test_close_price, length=lookback_window, batch_size=prediction_horizon)

list_of_models = ['build_LSTM_sentiment', 'build_LSTM_stacked_sentiment', 'build_LSTM_bidirectional_sentiment', 'build_GRU_sentiment', 'build_GRU_stacked_sentiment', 'build_GRU_bidirectional_sentiment']

output_df = {'Models': [], 'MSE':[], 'RMSE': [], 'MAPE': []}

for m in list_of_models:
    model = globals()[m](lookback_window, neurons, opt)
    history = model.fit(train_generator, epochs=epochs, verbose=0)
    prediction = model.predict(test_generator)
    y_pred = scaler.inverse_transform(prediction)

    MSE = mean_squared_error(y_test, y_pred, squared=True)
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    MAPE = Calc_MAPE(y_test, y_pred)
    output_df['Models'].append(m)
    output_df['MSE'].append(MSE)
    output_df['RMSE'].append(RMSE)
    output_df['MAPE'].append(MAPE)
    
naive_baseline_MSE = mean_squared_error(y_test, naive_prediction, squared=True)
naive_baseline_RMSE = mean_squared_error(y_test, naive_prediction, squared=False)
naive_baseline_MAPE = Calc_MAPE(y_test, y_pred)
output_df['Models'].append('naive_baseline')
output_df['MSE'].append(naive_baseline_MSE)
output_df['RMSE'].append(naive_baseline_RMSE)
output_df['MAPE'].append(naive_baseline_MAPE)

print(pd.DataFrame(output_df))