import sys
import os

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tuning_util import grid_search
from save_meta_data import save_meta_data
from datetime import datetime
import torch

ticker = "COST"
time_period = ["2019-01-01", "2025-01-01"]
param_grid = {
    "look_back": [50, 100, 150],
    "batch_size": [32, 64, 128],
    "input_dim": [1],
    "hidden_dim": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "output_dim": [1],
    "num_epochs": [30, 50, 70],
    "learning_rate": [0.01, 0.005, 0.001],
}

best_model, best_config, best_val_loss = grid_search(ticker, time_period, param_grid)

print(f"Best configuration: {best_config}")
print(f"Best validation loss: {best_val_loss}")

# Save the best model and its configuration
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output/{ticker}_{current_time}"
os.makedirs(output_dir, exist_ok=True)

torch.save(
    best_model.state_dict(), f"{output_dir}/{best_config['num_epochs']}_epochs_lstm.pth"
)
save_meta_data(
    output_dir,
    ticker,
    time_period,
    best_config["look_back"],
    best_config["batch_size"],
    best_config.pop("look_back", "batch_size"),
)
