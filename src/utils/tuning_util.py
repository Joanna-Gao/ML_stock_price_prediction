from itertools import product
from src.data_processing.data_prep import *
from src.model.LSTM import LSTM
from src.model.train import train_model


def grid_search(ticker: str, time_period: List[str], param_grid: dict):
    """
    Note: param_grid also contains look_back and batch_size
    """
    best_val_loss = np.inf
    best_config = None
    best_model = None

    for params in product(*param_grid.values()):
        config = dict(zip(param_grid.keys(), params))
        print(f"Training with config: {config}")
        train_loader, val_loader, data_after_split, data, scaler_obj = get_data(
            ticker, time_period, config["look_back"], config["batch_size"]
        )

        model = LSTM(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            output_dim=config["output_dim"],
        )
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        model, _, val_losses, _ = train_model(
            model,
            config,
            train_loader,
            val_loader,
            optimiser,
            loss_fn,
            config["look_back"],
            ticker,
            is_tuning=True,
        )

        print(f"Validation loss: {min(val_losses)}")

        if min(val_losses) < best_val_loss:
            print(
                f"Found new best configuration with validation loss: {min(val_losses)}"
            )
            best_val_loss = min(val_losses)
            best_config = config
            best_model = model

    return best_model, best_config, best_val_loss
