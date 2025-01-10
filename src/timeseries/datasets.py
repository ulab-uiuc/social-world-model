
import torch
from torch.utils.data import Dataset



class SeriesDataset(Dataset):

    def __init__(self, args, seq, data_type='train'):
        self.args = args
        self.seq = seq
        self.data_type = data_type
        self.max_len = args.max_len

    def __getitem__(self, index):

        seq_id = index
        items = self.seq[index]

        assert self.data_type in {"train", "valid", "test"}


        if self.data_type == "train":
            input_ids = items[:-3]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            answer = [items[-1]]



        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids

        input_ids = input_ids[-self.max_len:]

        assert len(input_ids) == self.max_len

        cur_tensors = (
            torch.tensor(seq_id, dtype=torch.long),  # seq_id for testing
            torch.tensor(input_ids, dtype=torch.float32),
            torch.tensor(answer, dtype=torch.float32),
        )

        return cur_tensors

    def __len__(self):
        return len(self.seq)


