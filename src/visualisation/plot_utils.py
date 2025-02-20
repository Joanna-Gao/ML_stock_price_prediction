import matplotlib.pyplot as plt


def plot_raw_stock_data(data, column):
    plt.plot(data.index, data[column])
    plt.xlabel('Date')
    plt.ylabel('Ave of High and Low Prices on a Given Day')
    plt.title('COST Stock Price')
    plt.show()
