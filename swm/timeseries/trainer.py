import torch.optim as optim

from swm.timeseries.models import AttentionModel, LSTMModel
from swm.timeseries.util import mse, rmse


class Trainer:
    def __init__(self, hidden_dim, lr, device, max_len, model_name):
        if model_name == 'LSTM':
            self.model = LSTMModel(1, hidden_dim, 1)  # Construct the model
        else:
            self.model = AttentionModel(
                1, hidden_dim, 1, max_len
            )  # Construct the model
        self.model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = mse

    # Train
    def train(self, input, label):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, label)
        loss.backward()
        self.optimizer.step()
        rmse_val = rmse(output, label)
        return loss.item(), rmse_val.item()

    # Evaluation
    def eval(self, input, label):
        self.model.eval()
        output = self.model(input)
        loss = self.loss(label, output)
        rmse_val = rmse(label, output)
        return loss.item(), rmse_val.item()
