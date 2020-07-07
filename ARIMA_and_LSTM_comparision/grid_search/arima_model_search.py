import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from work_on_file.output_to_file import *
import time

def evaluate_arima_model(series, orders ):
    start = time.time()
    if log == True:
        series = pd.Series(np.log(series), index=series.index)

    size = int(len(series) * data_split)
    train, test = series[0:size], series[size:len(series)]
    history = [val for val in train]
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(params[0], params[1], params[2]))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat[0])
        obs = test[t]
        history.append(obs)

    future_forecast = model_fit.forecast(future_periods)[0]
    test_dates = test.index

    if log == True:
        predictions = np.exp(predictions)
        test = pd.Series(np.exp(test), index=test_dates)
        future_forecast = np.exp(future_forecast)

    predictions = pd.Series(predictions, index=test_dates)

    error = np.sqrt(mean_squared_error(predictions, test))
    print('Test RMSE: %.3f' % error)

    end = time.time()

    result_to_file(filepath, 'ARIMA model \n Test RMSE: %.3f' % error + '\n' + str(end - start))