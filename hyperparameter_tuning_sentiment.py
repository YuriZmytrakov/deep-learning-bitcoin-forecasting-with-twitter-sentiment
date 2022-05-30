# Imports
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, pointbiserialr, spearmanr, kendalltau
import itertools
from keras.models import Sequential
from keras.layers import Dense, LSTM
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

print("################## Check correlations ##################")

data = df_price[['target','sentiment']]

print(pearsonr(data['target'], data['sentiment']))
print(spearmanr(data['target'], data['sentiment']))
print(kendalltau(data['target'], data['sentiment']))
print(pointbiserialr(data['target'], data['sentiment']))

print("-------------- Shifted ---")
data['target'] = np.roll(data['target'], 1)

print(pearsonr(data['target'], data['sentiment']))
print(spearmanr(data['target'], data['sentiment']))
print(kendalltau(data['target'], data['sentiment']))
print(pointbiserialr(data['target'], data['sentiment']))

print("################## Remove extra fields ##################")
df_price.drop('target', axis=1, inplace=True)
df_price.drop('open', axis=1, inplace=True)

print("################## Tuning ##################")

def build_LSTM_sentiment(lookback_window, neurons, opt):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(lookback_window, 2)))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model

print("WARNING - TAKES 24+ HOURS TO COMPLETE!!!") 
import itertools
def model_configs():
	# define scope of configs
	neurons = range(10, 50, 10)
	opt = ['adam','sgd','rmsprop','ftrl']
	epochs = [10, 20, 30, 40, 50]
	prediction_horizon = [5, 10, 30, 60, 90]
	lookback_window = [25, 50, 75, 100]
	a = [neurons, opt, epochs, prediction_horizon, lookback_window]
	# create configs
	configs = list(itertools.product(*a))
	print('Total configs: %d' % len(configs))
	return configs

output_dic = {'Combination': [], 'RMSE': [], 'Predictions': []}

cfg_list = model_configs()
counter = 0
for config in cfg_list:
    counter = counter+1
    print("running" + str(counter))
    prediction_horizon = config[3] # i.e. predict last week's values
    n_steps = config[4] # look back window reducing dues to high computitional cost
    n_features = 1 # of features in the data

    # LSTM input is of the form (n_samples, n_steps, n_features)
    X_train_price = df_price['close'].values[:-prediction_horizon]
    X_test_price = df_price['close'].values[-(n_steps + prediction_horizon):]
    y_test = df_price['close'].values[-prediction_horizon:]
    X_train_sentiment = df_price['sentiment'].values[:-prediction_horizon].reshape(-1, 1)
    X_test_sentiment = df_price['sentiment'].values[-(n_steps + prediction_horizon):].reshape(-1, 1)

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
    train_generator = TimeseriesGenerator(X_train, Y_train, length=n_steps, batch_size=prediction_horizon)
    test_generator = TimeseriesGenerator(X_test, X_test_close_price, length=n_steps, batch_size=prediction_horizon)

    model = build_LSTM_sentiment(n_steps, config[0], config[1])
    history = model.fit(train_generator, epochs=config[2], verbose=0)
    prediction = model.predict(test_generator)
    y_pred = scaler.inverse_transform(prediction)

    y_pred = np.nan_to_num(y_pred)
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    output_dic['Combination'].append(config)
    output_dic['RMSE'].append(RMSE)
    output_dic['Predictions'].append(y_pred)
        
output_df = pd.DataFrame(output_dic)
output_df_sorted = output_df.sort_values(by='RMSE', ascending=True)
print(output_df_sorted)