"""
包含：
1. def text_to_ids: 将 token化的句子转化为每个 token在vocab中的索引
2. class LSTMClassifier：模型主体
3. def evaluate： 模型推理
"""
import torch
import torch.nn as nn
import argparse
import numpy as np

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

"""
用kernel size=1,2,3三个卷积核来提取特征，再做cross attention后输出分类
"""
class TFClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=5, dropout=0.04, use_glove=False,
                 glove_matrix=None, stride=1):
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
        self.dropout = nn.Dropout(dropout)


def predict(model, src_tokens, word_2_id, id_2_word, max_len=15, device=device):
    """
    真正的推理过程：一字一字生成，不使用教师迫导 (Teacher Forcing)
    src_tokens: 输入的数学题 Tensor (B, L)
    """
    model.eval()
    bos_id = word_2_id['BOS']
    eos_id = word_2_id['EOS']
    batch_size = src_tokens.size(0)

    with torch.no_grad():
        # 1. 编码器只跑一次
        #  model 有分别调用 encoder 和 decoder 的接口
        src_emb = model.pe(model.embedding(src_tokens))
        enc_out = model.encoder(src_emb)

        # 2. 解码器初始输入只有 BOS
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device) * bos_id

        for _ in range(max_len):
            # 得到当前已生成序列的 embedding
            dec_emb = model.pe(model.embedding(generated))

            # 运行解码器 (需要传入 enc_out 进行 Cross Attention)
            # 注意：这里的 model 调用需要根据你的 Transformer.forward 结构微调
            logits = model.decoder(dec_emb, enc_out)

            # 取最后一个时间步的输出作为下一个词的预测
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # 拼接预测结果
            generated = torch.cat([generated, next_token], dim=1)

    return generated


