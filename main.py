from src.data_processing.data_prep import *
from src.model.LSTM import LSTM
import matplotlib.pyplot as plt


def main():
    time_period = ['2010-01-01', '2025-01-01']
    data = load_raw_data(time_period)
    look_back = 500
    train_set, validation_set, test_set = split_data(data, look_back=look_back)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                 num_layers=num_layers, output_dim=output_dim)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print(model)
    print(model.parameters())

    loss_hist = np.zeros(num_epochs)
    seq_dim = look_back - 1

    for t in range(num_epochs):
        y_train_pred = model(train_set[0])

        loss = loss_fn(y_train_pred, train_set[1])
        if t % 10 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        loss_hist[t] = loss.item()

        optimiser.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation
        optimiser.step()  # update the parameters

    plt.plot(loss_hist, label='Training loss')
    plt.show()


if __name__ == '__main__':
    main()
