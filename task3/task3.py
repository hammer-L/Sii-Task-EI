import torch
import torch.nn as nn
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)


"""
数据生成
f"BOS {x} + {y} = {x+y} EOS"
f"BOS {x} - {y} = {x-y} EOS"
"""
import random
from utils import WordEmbedding, Tokenizer_Digitwise
from utils import Generate_data

emb_dim = 32

train_data, train_gt, test_data, test_gt = Generate_data(train_data_nums=10000, test_data_nums=2000)
# print (train_data[:2])
# print (train_gt[:2])

tokenizer = Tokenizer_Digitwise()
ids = [tokenizer.text2ids(sentence) for sentence in train_data]
max_len = max(len(s) for s in ids)
print (max_len)
print (len(tokenizer.vocab))

train_data_ids = tokenizer.pad_sequences([tokenizer.text2ids(sentence) for sentence in train_data])
train_gt_ids = tokenizer.pad_sequences([tokenizer.text2ids(sentence) for sentence in train_gt])
test_data_ids = tokenizer.pad_sequences([tokenizer.text2ids(sentence) for sentence in test_data])
test_gt_ids = tokenizer.pad_sequences([tokenizer.text2ids(sentence) for sentence in test_gt])
# print (train_data_ids[:2])
# print (train_gt_ids[:2])

from model import Transformer, Decoder
from utils import predict, evaluate
from torch.utils.data import DataLoader
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

input_dim = 32
hidden_dim = input_dim
num_heads = 4
num_blocks = 4
vocab_size = len(tokenizer.vocab)
dropout = 0.1
batch_size = 64
epoches = 200
lr = 1e-3


def train(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_blocks=num_blocks, vocab_size=vocab_size,
          dropout=dropout,
          epoches=epoches, batch_size=batch_size, model_type='transformer'):
    log_dir = f"runs/sub1-{model_type}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除旧日志，避免曲线混淆
    writer = SummaryWriter(log_dir)

    train_dataset = [(torch.tensor(x), torch.tensor(y)) for x, y in zip(train_data_ids, train_gt_ids)]
    test_dataset = [(torch.tensor(x), torch.tensor(y)) for x, y in zip(test_data_ids, test_gt_ids)]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_type == "transformer":
        model = Transformer(input_dim, hidden_dim, num_heads, num_blocks, vocab_size, dropout=dropout).to(device)
    elif model_type == "decoder":
        model = Decoder(input_dim, hidden_dim, num_heads, num_blocks, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    print("start training")
    for epoch in range(epoches):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            enc_in = x
            dec_in = y[:, :-1]  # 去除最后一位 EOS
            target = y[:, 1:]  # 去除第一位 BOS，保持长度一致并且错位堆叠

            optimizer.zero_grad()
            logits = model(enc_in, dec_in)
            B, L, V = logits.shape

            # logits = (B, L, V) -> (B*L, V) 展开成二维
            # target = (B, L, 1) -> (B*L) 展成一位
            loss = criterion(logits.view(B * L, V), target.reshape(-1))
            loss.backward()
            optimizer.step()

            pred = logits.argmax(-1)
            mask = target != 0
            correct += (pred[mask] == target[mask]).sum().item()
            total += mask.sum().item()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        train_acc = 100 * correct / total

        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
        # print (f'train_loss: {train_loss}, train_acc: {train_acc}')

        if (epoch + 1) % 10 == 0:
            print(f"{epoch + 1}:")
            test_acc, results = evaluate(model, test_dataloader, device)
            writer.add_scalar('Accuracy/Test', test_acc, epoch + 1)

            sample_to_show = 5
            if sample_to_show > 0:
                for res in results[:sample_to_show]:
                    print(f"{res['pred']:<16} | {res['true']:<16} | {res['status']}")
                    sample_to_show -= 1

    return model


train()
