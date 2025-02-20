from src.data_processing.data_prep import *
from src.model.LSTM import LSTM
from src.model.train import train_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def main():
    ticker = "COST"
    time_period = ["2001-01-01", "2025-01-01"]
    data = load_raw_data(ticker, time_period)

    # normalise the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data["Ave"] = scaler.fit_transform(data["Ave"].values.reshape(-1, 1))

    batch_size = 64
    look_back = 100
    train_loader, val_loader, test_loader, original_data = split_data(
        data, look_back, batch_size
    )

    print("Obtained the data, creating model...")

    model_config = {
        "input_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "output_dim": 1,
        "num_epochs": 50,
    }

    model = LSTM(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        output_dim=model_config["output_dim"],
    )
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Model created, starting training...")

    model, train_losses, val_losses = train_model(
        model,
        model_config,
        train_loader,
        val_loader,
        optimiser,
        loss_fn,
        look_back,
        ticker,
    )

    print("Training complete, plotting the results...")

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")

    x_test_tensor = to_torch_tensor(original_data["x_test"])
    y_test_pred = model(x_test_tensor)
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(original_data["y_test"])

    figrue, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(data.index[-len(y_test_pred) :], y_test_pred, label="Predictions")
    axes.plot(data.index[-len(y_test) :], y_test, label="True Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
