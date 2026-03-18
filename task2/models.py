"""
包含：
1. def text_to_ids: 将 token化的句子转化为每个 token在vocab中的索引
2. class LSTMClassifier：模型主体
3. def evaluate： 模型推理
"""
import torch
import torch.nn as nn
import numpy as np
import math
device = "cuda0" if torch.cuda.is_available() else "cpu"
glove_path = "glove/glove.2024.wikigiga.50d/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

def text_to_ids(tokenized_texts, vocab, max_len=50):
    ids = []
    for sentence in tokenized_texts:
        sentence_ids = [vocab.get(word, 1) for word in sentence]

        # truncate or padding
        if len(sentence_ids) > max_len:
            sentence_ids = sentence_ids[:max_len]
        else:
            sentence_ids += [0] * (max_len - len((sentence_ids)))
        ids.append(sentence_ids)
    return torch.tensor(ids)


def build_glove_matrix(vocab, glove_path, embedding_dim=50):
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(
        0, 0.6, (vocab_size, embedding_dim)
    ).astype(np.float32)

    # PAD 置零（假设 PAD=0）
    embedding_matrix[0] = np.zeros(embedding_dim)

    print("Loading GloVe...")

    # 只加载 vocab 交集部分
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()

            if len(values) < embedding_dim + 1:
                continue

            # 最后 50 个一定是向量
            vector_part = values[-embedding_dim:]

            try:
                vector = np.asarray(vector_part, dtype='float32')
            except ValueError:
                continue

            # 前面全部拼成 token
            word = " ".join(values[:-embedding_dim])

            if word not in vocab:
                continue

            idx = vocab[word]
            embedding_matrix[idx] = vector

    print("GloVe loaded.")
    return torch.tensor(embedding_matrix)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=5, n_layers=1, dropout=0.04, use_glove=False, glove_matrix=None):
        super().__init__()

        if not use_glove:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                glove_matrix,
                freeze=True,
                padding_idx=0
            )
        self.use_glove = use_glove
        self.glove_matrix = glove_matrix

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers>1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x : (b, l) 15位长的为idx
        """
        emb = self.embedding(x)
        emb = self.dropout(emb)    # emb: (b, l, emb_dim)

        # 2. LSTM
        # output: (Batch, Seq_Len, Hidden_Dim)
        # hidden: (Num_Layers, Batch, Hidden_Dim)
        # cell: (Num_Layers, Batch, Hidden_Dim)
        output, hidden = self.rnn(emb)

        # 提取最后一层的特征
        last_hidden = output[:, -1, :]
        x = self.fc1(self.dropout(last_hidden))
        x = torch.relu(x)
        logits = self.fc2(self.dropout(x))

        return logits


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=5, dropout=0.04, use_glove=False,
                 glove_matrix=None, kernel_size=3, stride=1):
        super().__init__()

        if not use_glove:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                glove_matrix,
                freeze=True,
                padding_idx=0
            )
        self.use_glove = use_glove
        self.glove_matrix = glove_matrix

        # input = (B, emb_dim, L) -> output = (B, hidden_dim, L)
        self.cnn1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            padding=(kernel_size-1)//2
        )
        self.cnn2 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            padding=(kernel_size - 1) // 2
        )

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = self.cnn1(x)
        x = torch.relu(x)
        x = torch.max(x, dim=2)[0] # global max pooling, 在embedding 维度上找到最大值
        x = self.fc1(self.dropout(x))
        x = torch.relu(x)
        logits = self.fc2(self.dropout(x))

        return logits

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

"""
self, vocab_size, embedding_dim, hidden_dim, num_classes=5, dropout=0.04, use_glove=False,
                 glove_matrix=None, kernel_size=3, stride=1
"""
class TransformerClassifier(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, num_classes=5, use_glove=False, glove_matrix=None,
                 nhead=2,num_layers=1, dropout=0.1,max_len=52):
        super().__init__()

        if not use_glove:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                glove_matrix,
                freeze=True,
                padding_idx=0
            )

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        # x shape
        # (batch, seq_len)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def evaluate(model, dataloader, device, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_x, batch_y in dataloader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            total_loss += loss.item()

            _, pred = torch.max(outputs, dim=1)

            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100 * correct / total

    return avg_loss, acc

