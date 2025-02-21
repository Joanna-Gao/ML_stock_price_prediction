import sys
import os
import argparse
import matplotlib.pyplot as plt

from src.data_processing.data_prep import *
from src.model.LSTM import LSTM
from src.model.train import train_model
from src.utils.save_meta_data import save_meta_data
from src.visualisation.plot_utils import *


def get_args():
    parser = argparse.ArgumentParser(description="LSTM model for stock prediction")
    parser.add_argument(
        "--ticker", type=str, default="COST", help="Ticker of the stock to predict"
    )
    args = parser.parse_args()
    return args


def main(ticker: str):
    time_period = ["2019-01-01", "2025-01-01"]
    batch_size = 64
    look_back = 100

    train_loader, val_loader, data_after_split, data, scaler_obj = get_data(
        ticker, time_period, look_back, batch_size
    )

    print("Obtained the data, creating model...")

    model_config = {
        "input_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "output_dim": 1,
        "num_epochs": 50,
        "learning_rate": 0.01,
    }

    model = LSTM(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        output_dim=model_config["output_dim"],
    )
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])

    print("Model created, starting training...")

    model, train_losses, val_losses, output_dir = train_model(
        model,
        model_config,
        train_loader,
        val_loader,
        optimiser,
        loss_fn,
        look_back,
        ticker,
    )

    save_meta_data(output_dir, ticker, time_period, look_back, batch_size, model_config)

    print("Training complete, plotting the results...")

    plot_train_val_losses(train_losses, val_losses)

    x_test_tensor = to_torch_tensor(data_after_split["x_test"])
    y_test_pred = model(x_test_tensor)
    y_test_pred = scaler_obj.inverse_transform(y_test_pred.detach().numpy())
    y_true = scaler_obj.inverse_transform(data["Ave"].values.reshape(-1, 1))

    plot_predictions(data, y_test_pred, y_true)


if __name__ == "__main__":
    config = get_args()
    main(config.ticker)
