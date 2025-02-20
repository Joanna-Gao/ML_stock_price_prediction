from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
import torch


def load_raw_data(ticker: str, time_period: List[str],
                  save_data: bool = False) -> pd.DataFrame:
    try:
        cost = yf.Ticker(ticker)
    except Exception as e:
        print(f"Error: {e}")
        return
    data = cost.history(start=time_period[0], end=time_period[1])

    # the index for the df is date and time data in NY timezone, normalise it
    data.index = data.index.tz_localize(None)

    # calculate a mean to use as the stock price everyday
    data['Ave'] = (data['High'] + data['Low']) / 2

    # remove useless columns
    data = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume',
                              'Dividends', 'Stock Splits'], axis=1)

    if save_data:
        data.to_csv('costco_ave_stock_price_%s_to_%s'
                    % (time_period[0], time_period[1]))

    return data


def split_data(price: pd.DataFrame, look_back: int,
               save_data: bool = False) -> tuple:

    data_raw = price.values
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index:index+look_back])

    data = np.array(data)
    # train, validation, test split = 70:15:15
    train_size = int(0.7 * data.shape[0])
    validation_end = int(0.85 * data.shape[0])

    x_train = data[:train_size, :-1, :]
    y_train = data[:train_size, -1, :]

    x_validation = data[train_size:validation_end, :-1, :]
    y_validation = data[train_size:validation_end, -1, :]

    x_test = data[validation_end:, :-1, :]
    y_test = data[validation_end:, -1, :]

    # train_set, validation_set, test_set = (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

    # if save_data:
    #     # save the sets
    #     np.savez('costco_stock_data_ml_ready.npz', train_set=train_set,
    #              validation_set=validation_set, test_set=test_set)

    # make data into torch tensors
    def to_torch_tensor(data):
        return torch.from_numpy(data).type(torch.Tensor)

    train_set = (to_torch_tensor(x_train), to_torch_tensor(y_train))
    validation_set = (to_torch_tensor(x_validation),
                      to_torch_tensor(y_validation))
    test_set = (to_torch_tensor(x_test), to_torch_tensor(y_test))

    return train_set, validation_set, test_set
