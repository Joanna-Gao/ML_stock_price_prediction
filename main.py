from src.data_processing.data_prep import *
from src.model.LSTM import LSTM
from src.model.train import train_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def main():
    ticker = "IBM"
    time_period = ['2010-01-02','2017-10-11']
    data = load_raw_data(ticker, time_period)
    
    # normalise the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Ave'] = scaler.fit_transform(data['Ave'].values.reshape(-1, 1))
    
    look_back = 500
    train_set, validation_set, test_set = split_data(data, look_back=look_back)

    print("Obtained the data, creating model...")

    model_config = {'input_dim': 1, 'hidden_dim': 32, 'num_layers': 2, 
                    'output_dim': 1, 'num_epochs': 100}

    model = LSTM(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        output_dim=model_config['output_dim'],
    )
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Model created, starting training...")

    model, loss_hist = train_model(model, model_config, train_set, optimiser, 
                                   loss_fn, look_back, ticker)

    print("Training complete, plotting the results...")

    plt.plot(loss_hist, label="Training loss")

    y_test_pred = model(test_set[0])
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(test_set[1].detach().numpy())

    figrue, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(data.index[-len(y_test_pred) :], y_test_pred, label="Predictions")
    axes.plot(data.index[-len(y_test) :], y_test, label="True Price")

    plt.show()


# if __name__ == "__main__":
#     main()
main()