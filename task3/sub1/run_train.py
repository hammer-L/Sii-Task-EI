import torch
import torch.nn as nn
import random
from test3 import generate_dataset, TransformerModel, DecoderModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 20000
# 数据生成
train_x, train_y, test_x, test_y = generate_dataset(n_samples=n)

tokens = ['0','1','2','3','4','5','6','7','8','9','+', '=','<PAD>','<BOS>','<EOS>']
vocab_size = len(tokens)

s2i = {c: i for i, c in enumerate(tokens)}  # 从token到ids
i2s = {i: c for c, i in s2i.items()}  # 从ids到token
# print (s2i)
# print (i2s)
PAD = s2i['<PAD>']
BOS = s2i['<BOS>']
EOS = s2i['<EOS>']

test_sample = ['123+456']


def tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i] == "<":
            j = s.index(">", i)
            tokens.append(s[i:j + 1])
            i = j + 1
        else:
            tokens.append(s[i])
            i += 1
    return tokens


def encode_tokens(tokens):
    return [s2i[t] for t in tokens]


def pad(seq, length):
    return seq + [PAD] * (length - len(seq))


def build_batch(batch_size, data_x, data_y):
    xs = []
    ys_in = []
    ys_out = []

    idxs = random.sample(range(len(data_x)), batch_size)

    for i in idxs:
        x = data_x[i]
        y_in_str, y_out_str = data_y[i]

        # encoder
        x = encode_tokens(tokenize(x))

        # decoder input
        y_in = encode_tokens(tokenize(y_in_str))

        # decoder target
        y_out = encode_tokens(tokenize(y_out_str))

        xs.append(pad(x, 11))
        ys_in.append(pad(y_in, 8))
        ys_out.append(pad(y_out, 8))

    return (
        torch.tensor(xs).to(device),
        torch.tensor(ys_in).to(device),
        torch.tensor(ys_out).to(device)
    )

def build_decoder_batch(batch_size, data_x, data_y):

    xs = []
    ys = []

    idxs = random.sample(range(len(data_x)), batch_size)

    for i in idxs:

        x_str = data_x[i]
        y_in_str, y_out_str = data_y[i]

        # 取答案字符串
        ans = y_out_str.replace("<EOS>", "")

        # 拼接完整序列
        seq = "<BOS>" + x_str + "=" + ans + "<EOS>"

        tokens = tokenize(seq)
        ids = encode_tokens(tokens)

        # input / target
        inp = ids[:-1]
        tgt = ids[1:]

        xs.append(pad(inp, 20))
        ys.append(pad(tgt, 20))

    return (
        torch.tensor(xs).to(device),
        torch.tensor(ys).to(device)
    )

def mask_question_loss(x, y):
    """
    x : input tokens
    y : target tokens
    把 '=' 之前的 target 变成 PAD
    """
    eq_id = s2i["="]

    for i in range(x.size(0)):
        pos = (x[i] == eq_id).nonzero(as_tuple=True)[0]

        if len(pos) > 0:
            eq_pos = pos[0].item()
            y[i, :eq_pos+1] = PAD   # '=' 之前全部忽略

    return y



"""
training
"""
import os
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(lr, vocab_size, batch_size, epochs, model_type="transformer"):
    if model_type=="transformer":
        log_dir = f"runs/sub1/{model_type}(lr={lr})"
        model=TransformerModel(vocab_size=vocab_size).to(device)
    if model_type == "decoder-only":
        log_dir = f"runs/sub1/{model_type}(lr={lr})"
        model = DecoderModel(vocab_size=vocab_size).to(device)

    # tensorboad 初始化
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  # 删除旧日志，避免曲线混淆
    writer = SummaryWriter(log_dir)
    
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    criterion=nn.CrossEntropyLoss(ignore_index=PAD)
    
    print ("start training!")
    for epoch in tqdm(range(epochs)):
        # 记载数据
        if model_type == "transformer":
            x, y_in, y_out = build_batch(batch_size, train_x, train_y)
            logits = model(x, y_in)
            target = y_out
        else:  # decoder-only
            x, y = build_decoder_batch(batch_size, train_x, train_y)
            logits = model(x)
            target = mask_question_loss(x, y) # 增加mask， 把 ‘=’ 前的loss都忽略掉
    
        # print("logits shape:", logits.shape) # [128, 8, 14]
        # print("y_out shape:", y_out.shape) # [128, 8]
        # print("vocab_size:", vocab_size) # 14
    
        loss = criterion(
            logits.view(-1,vocab_size),
            target.view(-1)
        )
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/Train', loss.item(), epoch+1)

        if (epoch+1)%200==0:
            if model_type == "transformer":
                train_acc = evaluate(train_x, train_y, 200)
                test_acc = evaluate(test_x, test_y, 200)
            else:
                train_acc = evaluate_decoder(train_x, train_y, model, 200)
                test_acc = evaluate_decoder(test_x, test_y, model, 200)

            writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
            writer.add_scalar('Accuracy/Test', test_acc, epoch+1)
        
        if (epoch + 1) % 5000 == 0:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, f"ckpts/{model_type}_lr{lr}_epoch_{epoch+1}.pt")

def predict(x_str, model):
    model.eval()

    # encoder 输入
    x = encode_tokens(tokenize(x_str))
    x = torch.tensor([pad(x, 11)]).to(device)

    y = [BOS]
    for _ in range(7):
        y_tensor = torch.tensor([pad(y, 8)]).to(device)
        with torch.no_grad():
            logits = model(x, y_tensor)
        next_token = logits[0, len(y) - 1].argmax().item()
        if next_token == EOS:
            break
        y.append(next_token)
    # 转回字符串
    result = "".join(i2s[i] for i in y[1:])
    return result


def evaluate(test_x, test_y, model, n=100):
    correct = 0

    # 随机打乱idx
    idxs = random.sample(range(len(test_x)), n)
    for i in idxs:
        x = test_x[i]
        pred = predict(x, model=model)
        true = test_y[i][1].replace("<EOS>", "")
        if pred == true:
            correct += 1
    return correct / n

def predict_decoder(expr, model):
    model.eval()
    seq = "<BOS>" + expr + "="
    tokens = tokenize(seq)
    ids = encode_tokens(tokens)
    
    for _ in range(10):
        x = torch.tensor([pad(ids, 20)]).to(device)
        with torch.no_grad():
            logits = model(x)
        next_token = logits[0, len(ids)-1].argmax().item()
        if next_token == EOS:
            break
        ids.append(next_token)

    result = "".join(i2s[i] for i in ids)
    return result.split("=")[1]

def evaluate_decoder(data_x, data_y, model, n=200):
    idxs = random.sample(range(len(data_x)), n)
    correct = 0
    for i in idxs:
        x = data_x[i]
        pred = predict_decoder(x, model)
        true = data_y[i][1].replace("<EOS>", "")
        if pred == true:
            correct += 1
    return correct / n



lr = 3e-4
vocab_size = len(tokens)
batch_size = 2048*2
epochs = 25000


train(model_type="decoder-only", lr=lr, vocab_size=vocab_size, batch_size=batch_size, epochs=epochs)
