import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import time
from work_on_file.output_to_file import *
def create_dataset(data_series, look_back, split_frac, transforms):
    if transforms[0] == True:
        dates = data_series.index
        data_series = pd.Series(np.log(data_series), index=dates)

    if transforms[1] == True:
        dates = data_series.index
        data_series = pd.Series(data_series - data_series.shift(1), index=dates).dropna()

    dates = data_series.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
    data_series = pd.Series(scaled_data[:, 0], index=dates)

    df = pd.DataFrame()
    for i in range(look_back + 1):
        label = ''.join(['t-', str(i)])
        df[label] = data_series.shift(i)
    df = df.dropna()
    print(df.tail())

    size = int(split_frac * df.shape[0])
    train = df[:size]
    test = df[size:]

    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    train_dates = train.index

    X_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    test_dates = test.index

    X_train = np.reshape(X_train, (X_train.shape[0], 1, look_back))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, look_back))

    return X_train, y_train, X_test, y_test, train_dates, test_dates, scaler, transforms


def inverse_transforms(train_predict, y_train, test_predict, y_test, data_series, train_dates, test_dates, scaler, transforms):
    train_predict = pd.Series(scaler.inverse_transform(train_predict.reshape(-1, 1))[:, 0], index=train_dates)
    y_train = pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1))[:, 0], index=train_dates)

    test_predict = pd.Series(scaler.inverse_transform(test_predict.reshape(-1, 1))[:, 0], index=test_dates)
    y_test = pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0], index=test_dates)

    if (transforms[1] == True) & (transforms[0] == True):
        train_predict = pd.Series(train_predict + np.log(data_series.shift(1)), index=train_dates).dropna()
        y_train = pd.Series(y_train + np.log(data_series.shift(1)), index=train_dates).dropna()

        test_predict = pd.Series(test_predict + np.log(data_series.shift(1)), index=test_dates).dropna()
        y_test = pd.Series(y_test + np.log(data_series.shift(1)), index=test_dates).dropna()

    elif transforms[1] == True:
        train_predict = pd.Series(train_predict + data_series.shift(1), index=train_dates).dropna()
        y_train = pd.Series(y_train + data_series.shift(1), index=train_dates).dropna()

        test_predict = pd.Series(test_predict + data_series.shift(1), index=test_dates).dropna()
        y_test = pd.Series(y_test + data_series.shift(1), index=test_dates).dropna()

    if transforms[0] == True:
        train_predict = pd.Series(np.exp(train_predict), index=train_dates)
        y_train = pd.Series(np.exp(y_train), index=train_dates)

        test_predict = pd.Series(np.exp(test_predict), index=test_dates)
        y_test = pd.Series(np.exp(y_test), index=test_dates)
    return train_predict, y_train, test_predict, y_test


def lstm_model(data_series, look_back, split, transforms, lstm_params, lr, num_of_layers, filepath):
    np.random.seed(1)
    start = time.time()
    X_train, y_train, X_test, y_test, train_dates, test_dates, scaler, transforms = create_dataset(data_series, look_back, split, transforms)
    model = Sequential()
    model.add(LSTM(lstm_params[0], input_shape=(1, look_back), name='layer_0'))
    for i in range(num_of_layers-1):
        model.add(Reshape((1, lstm_params[0])))
        model.add(LSTM(lstm_params[0], name='layer_'+str(i+1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
    model.fit(X_train, y_train, epochs=lstm_params[1], batch_size=1, verbose=lstm_params[2])

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict, y_train, test_predict, y_test = \
        inverse_transforms(train_predict, y_train, test_predict, y_test, data_series, train_dates, test_dates, scaler, transforms)

    error = np.sqrt(mean_squared_error(train_predict, y_train))
    print('Train RMSE: %.3f' % error)
    error = np.sqrt(mean_squared_error(test_predict, y_test))
    print('Test RMSE: %.3f' % error)
    end = time.time()
    result_to_file(filepath, 'LSTM model \n Test RMSE: %.3f' % error + '\n' + str(end-start))
    return train_predict, y_train, test_predict, y_test