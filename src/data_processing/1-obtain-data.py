'''
This file deals with the raw data, specifically it does the following:
- obtains data from Yahoo finance API through the yfinance package
- Initial data cleaning
- Exploratory plotting to examine the data
'''
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def main():
    cost = yf.Ticker('COST')  # using costco because I love costco lol
    data = cost.history(start='2010-01-01', end='2025-01-01')  # it's now a df

    # the index for the df is date and time data in NY timezone, normalise it
    data.index = data.index.tz_localized(None)

    # calculate a mean to use as the stock price everyday
    data['Ave'] = (data['High'] + data['Low']) / 2

    # remove useless columns
    data = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume',
                              'Dividends', 'Stock Splits'], axis=1)

    # exploratory data analysis: plotting some basic plots to see trends
    plt.plot(data.index, data['Ave'])
    plt.xlabel('Date')
    plt.ylabel('Ave of High and Low Prices on a Given Day')
    plt.title('COST Stock Price')
    plt.show()

    # storing the stock data
    start = data.index[0].strftime('%Y-%m-%d')
    end = data.index[-1].strftime('%Y-%m-%d')
    data.to_csv('costco_ave_stock_price_%s_to_%s' % (start, end))


if __name__ == '__main__':
    main()
