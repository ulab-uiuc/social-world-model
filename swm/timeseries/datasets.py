
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
            target_pos = items[1:-2]

            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]



        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len

        cur_tensors = (
            torch.tensor(seq_id, dtype=torch.long),  
            torch.tensor(input_ids, dtype=torch.float32),
            torch.tensor(target_pos, dtype=torch.float32),
            torch.tensor(answer, dtype=torch.float32),
        )

        return cur_tensors

    def __len__(self):
        return len(self.seq)


