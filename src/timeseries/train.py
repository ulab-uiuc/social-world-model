import numpy as np
import torch
from trainer import Trainer
from util import DataLoader
import matplotlib.pyplot as plt
import argparse
import json
from datasets import SeriesDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import tqdm

# Hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int,
                    default=32)
parser.add_argument('--max_len', type=int,
                    default=120)
parser.add_argument('--epochs', type=int,
                    default=50)
parser.add_argument('--batch_size', type=int,
                    default=16)
parser.add_argument('--lr', type=float,
                default=0.0001)
parser.add_argument('--dataset', type=str,
                    default="polymarket")
parser.add_argument('--model_name', type=str,
                    default="LSTM")
args = parser.parse_args()
data = []

if args.dataset == "polymarket":
    file_name = "../../data/polymarket_data.jsonl"
    with open(file_name, 'r') as fcc_file:
        for line in fcc_file:
            obj = json.loads(line)
            for m in obj['markets']:
                if len(m['history'].keys()) == 0: continue
                k = list(m['history'].keys())[-1]
                if m['history'][k] is None or len(m['history'][k]) <= 5: continue
                newlist = sorted(m['history'][k], key=lambda d: d['t'])
                data.append([e['p'] for e in newlist])
else:
    file_name = "../../data/xx.jsonl"
    with open(file_name, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)

    for k in fcc_data.keys():
        data.append(list(fcc_data[k]['citations'].values()))

    

train_dataset = SeriesDataset(args, data, data_type='train')
eval_dataset = SeriesDataset(args, data, data_type='valid')

dataloader = {}

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
dataloader['train_loader'] = train_dataloader

eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)


epochs = args.epochs
device = 'cuda:0'
device = torch.device(device)

# Intialize the model and training process
trainer = Trainer(hidden_dim = args.hidden_dim, lr = args.lr, device = device, max_len = args.max_len, model_name = args.model_name)    

train_loss = []
test_loss = []



# Train the model
for epoch in range(1, epochs + 1):

    pretrain_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                desc=f"Train {args.model_name}-{args.dataset} Epoch:{epoch}",
                                total=len(train_dataloader),
                                bar_format="{l_bar}{r_bar}")
    train_rmse = []

    for i, batch in pretrain_data_iter:
        # 0. batch_data will be sent into the device(GPU or CPU)
        batch = tuple(t.to(device) for t in batch)
        _, trainx, trainy = batch
        metrics = trainer.train(trainx.unsqueeze(-1), trainy.unsqueeze(-1))
        train_rmse.append(metrics[1])
    train_mean_rmse = np.mean(train_rmse)
    train_loss.append(train_mean_rmse)

    valid_data_iter = tqdm.tqdm(enumerate(eval_dataloader),
                                desc=f"Eval {args.model_name}-{args.dataset} Epoch:{epoch}",
                                total=len(eval_dataloader),
                                bar_format="{l_bar}{r_bar}")
    test_rmse = []

    for i, batch in valid_data_iter:
        # 0. batch_data will be sent into the device(GPU or CPU)
        batch = tuple(t.to(device) for t in batch)
        _, testx, testy = batch
        metrics = trainer.eval(testx.unsqueeze(-1), testy.unsqueeze(-1))
        test_rmse.append(metrics[1])
    test_mean_rmse = np.mean(test_rmse)
    test_loss.append(test_mean_rmse)
    # Evaluate via MSE and RMSE
    # test_rmse = []
    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testy = torch.Tensor(y).to(device)
    #     metrics = trainer.eval(testx.unsqueeze(-1), testy.unsqueeze(-1))
    #     test_rmse.append(metrics[1])

    # test_mean_emse = np.mean(test_rmse)
    # test_loss.append(test_mean_emse)

# Plot the RMSE loss
plt.plot(train_loss, label="train")
plt.plot(test_loss, color='red', label="test")
print(train_loss)
print(test_loss)
plt.legend()
plt.ylabel("RMSE")
plt.xlabel("Iteration")
plt.savefig(f"output/loss_{args.model_name}_{args.hidden_dim}_{args.lr}.pdf")
