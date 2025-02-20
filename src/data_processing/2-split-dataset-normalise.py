'''
This file prepares the cleaned data for machine learning input. The data is 
splited into training, validation and testing sets, normalised and saved
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def main():
    price = pd.read_csv('costco_ave_stock_price_2010-01-04_to_2025-01-02')

    # reshape the price so that it's in shape (n, 1)
    price = price.reshape(-1, 1)

    # normalise the price (make them range between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaled = scaler.fit_transform(price)

    # split the price into training, validation and testing sets
    # current ratio is 70%:15%:15%
    train_split = int(0.7*len(price_scaled))
    validation_split = int(0.85*len(price_scaled))
    train_set, validation_set, test_set = np.split(price_scaled, 
                                                   [train_split, validation_split])

    # save the sets
    np.savez('costco_stock_price_ml_ready.npz', train_set=train_set, 
             validation_set=validation_set, test_set=test_set)

    


if __name__ == '__main__':
    main()
