import matplotlib.pyplot as plt


def plot_raw_stock_data(data, column):
    plt.plot(data.index, data[column])
    plt.xlabel("Date")
    plt.ylabel(f"{column} of High and Low Prices on a Given Day")
    plt.title("Stock Price")
    plt.show()


def plot_train_val_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_predictions(data, y_test_pred, y_true):
    figrue, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(data.index[-len(y_test_pred) :], y_test_pred, label="Predictions")
    axes.plot(data.index, y_true, ".", label="True Price")
    plt.legend()
    plt.show()
