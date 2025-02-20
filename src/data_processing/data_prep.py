import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(save_data=False):
    cost = yf.Ticker('COST')  # using costco because I love costco lol
    data = cost.history(start='2010-01-01', end='2025-01-01')  # it's now a df

    # the index for the df is date and time data in NY timezone, normalise it
    data.index = data.index.tz_localized(None)

    # calculate a mean to use as the stock price everyday
    data['Ave'] = (data['High'] + data['Low']) / 2

    # remove useless columns
    data = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume',
                              'Dividends', 'Stock Splits'], axis=1)

    if save_data:
        start = data.index[0].strftime('%Y-%m-%d')
        end = data.index[-1].strftime('%Y-%m-%d')
        data.to_csv('costco_ave_stock_price_%s_to_%s' % (start, end))

    return data


def split_data(data, save_data=False):
    # reshape the data so that it's in shape (n, 1)
    data = data.reshape(-1, 1)

    # normalise the data (make them range between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # split the data into training, validation and testing sets
    # current ratio is 70%:15%:15%
    train_split = int(0.7*len(data_scaled))
    validation_split = int(0.85*len(data_scaled))
    train_set, validation_set, test_set = np.split(data_scaled,
                                                   [train_split, validation_split])

    if save_data:
        # save the sets
        np.savez('costco_stock_data_ml_ready.npz', train_set=train_set,
                 validation_set=validation_set, test_set=test_set)

    return train_set, validation_set, test_set
