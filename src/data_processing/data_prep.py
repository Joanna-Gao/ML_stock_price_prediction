from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


def load_raw_data(
    ticker: str, time_period: List[str], save_data: bool = False
) -> pd.DataFrame:
    try:
        cost = yf.Ticker(ticker)
    except Exception as e:
        print(f"Error: {e}")
        return
    data = cost.history(start=time_period[0], end=time_period[1])

    # the index for the df is date and time data in NY timezone, normalise it
    data.index = data.index.tz_localize(None)

    # calculate a mean to use as the stock price everyday
    data["Ave"] = (data["High"] + data["Low"]) / 2

    # normalise the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_obj = scaler.fit(data["Ave"].values.reshape(-1, 1))
    data["Ave"] = scaler_obj.transform(data["Ave"].values.reshape(-1, 1))

    # remove useless columns
    data = data.drop(
        columns=["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
        axis=1,
    )

    if save_data:
        data.to_csv(
            "costco_ave_stock_price_%s_to_%s" % (time_period[0], time_period[1])
        )

    return data, scaler_obj


def to_torch_tensor(data: np.ndarray) -> torch.Tensor:
    # make data into torch tensors
    return torch.from_numpy(data).type(torch.Tensor)


def create_loader(
    x_data: torch.Tensor, y_data: torch.Tensor, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataset = TensorDataset(to_torch_tensor(x_data), to_torch_tensor(y_data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_data(
    price: pd.DataFrame, look_back: int, batch_size: int, save_data: bool = False
) -> (DataLoader, DataLoader, DataLoader, dict):

    data_raw = price.values
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index : index + look_back])

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

    # Save the original datasets for plotting
    data_after_split = {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": x_test,
        "y_test": y_test,
    }

    # train_set, validation_set, test_set = (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

    # if save_data:
    #     # save the sets
    #     np.savez('costco_stock_data_ml_ready.npz', train_set=train_set,
    #              validation_set=validation_set, test_set=test_set)

    train_loader = create_loader(x_train, y_train, batch_size=batch_size)
    validation_loader = create_loader(x_validation, y_validation, batch_size=batch_size)
    # test_loader = create_loader(x_test, y_test, batch_size=batch_size)

    return train_loader, validation_loader, data_after_split


def get_data(ticker: str, time_period: List[str], look_back: int, batch_size: int):
    original_data, scaler_obj = load_raw_data(ticker, time_period)

    train_loader, val_loader, data_after_split = split_data(
        original_data, look_back, batch_size
    )

    return train_loader, val_loader, data_after_split, original_data, scaler_obj
