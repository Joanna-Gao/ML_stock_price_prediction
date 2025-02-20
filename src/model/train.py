import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def train_model(
    model: nn.Module,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    look_back: int,
    ticker: str = "",
) -> (nn.Module, np.ndarray):

    train_losses = np.zeros(config["num_epochs"])
    val_losses = np.zeros(config["num_epochs"])

    for epoch in range(config["num_epochs"]):
        #### ----------------- 1. TRAINING PHASE  ----------------- ####
        model.train()
        running_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            # forward pass
            y_train_pred = model(x_batch)
            loss = loss_fn(y_train_pred, y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_train_loss += loss.item()

        # averge training loss over all batches
        train_loss_epoch = running_train_loss / len(train_loader)
        train_losses[epoch] = train_loss_epoch

        #### ----------------- 2. VALIDATION PHASE ----------------- ####
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                y_val_pred = model(x_val)
                loss = loss_fn(y_val_pred, y_val)
                running_val_loss += loss.item()

        # average validation loss over all batches
        val_loss_epoch = running_val_loss / len(val_loader)
        val_losses[epoch] = val_loss_epoch

        #### ----------------- 3. LOGGING / PRINTING ---------------- ####
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{config['num_epochs']}] "
                f"Train Loss: {train_loss_epoch:.4f}, "
                f"Val Loss: {val_loss_epoch:.4f}"
            )

    torch.save(
        model.state_dict(), f"output/{ticker}_{config['num_epochs']}_epochs_lstm.pth"
    )
    np.save(
        f"output/{ticker}_{config['num_epochs']}_epochs_train_loss_hist.npy",
        train_losses,
    )
    np.save(
        f"output/{ticker}_{config['num_epochs']}_epochs_val_loss_hist.npy", val_losses
    )

    return model, train_losses, val_losses
