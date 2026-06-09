import numpy as np
import torch


# Self-defined data loader
class DataLoader(object):
    def __init__(self, x, y, batch_size):
        self.batch_size = batch_size
        self.cur_index = 0

        length_pad = (batch_size - (len(x) % batch_size)) % batch_size
        x_pad = np.repeat(x[-1:], length_pad, axis=0)
        y_pad = np.repeat(y[-1:], length_pad, axis=0)
        x = np.concatenate([x, x_pad], axis=0)
        y = np.concatenate([y, y_pad], axis=0)

        self.total_size = len(x)
        self.n_batch = int(self.total_size // self.batch_size)
        self.x = x
        self.y = y

    def shuffle(self):
        permutation = np.random.permutation(self.total_size)
        x, y = self.x[permutation], self.y[permutation]
        self.x = x
        self.y = y

    def get_iterator(self):
        self.cur_index = 0

        def _wrapper():
            while self.cur_index < self.n_batch:
                start_index = self.batch_size * self.cur_index
                end_index = min(self.total_size, self.batch_size * (self.cur_index + 1))
                yield (
                    self.x[start_index:end_index, ...],
                    self.y[start_index:end_index, ...],
                )
                self.cur_index += 1

        return _wrapper()


# MSE
def mse(output, label):
    mse_value = (output - label) ** 2
    return torch.mean(mse_value)


# RMSE
def rmse(output, label):
    mse_value = mse(output=output, label=label)
    return torch.sqrt(mse_value)
