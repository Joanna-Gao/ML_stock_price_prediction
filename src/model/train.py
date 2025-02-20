import torch
import torch.nn as nn
import numpy as np


def train_model(model: nn.Module,
                config: dict,
                train_set: tuple[torch.Tensor, torch.Tensor],
                optimiser: torch.optim.Optimizer,
                loss_fn: nn.Module,
                look_back: int,
                ticker: str = ''):
    
    loss_hist = np.zeros(config['num_epochs'])

    for epoch in range(config['num_epochs']):
        model.train()

        # forward pass
        y_train_pred = model(train_set[0])
        loss = loss_fn(y_train_pred, train_set[1])

        if epoch % 10 == 0 and epoch != 0:
           print(f'Epoch {epoch} MSE: {loss.item()}')

        loss_hist[epoch] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch.save(model.state_dict(), f'output/{ticker}_{config['num_epochs']}_epochs_lstm.pth')
    np.save(f'output/{ticker}_{config['num_epochs']}_epochs_loss_hist.npy', loss_hist)

    return model, loss_hist
