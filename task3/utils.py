import torch
from torch import nn
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Generate_data(train_data_nums=5000, test_data_nums=1000):
    train_data = []
    train_gt = []
    test_data = []
    test_gt = []

    random.seed(42)
    for _ in range(train_data_nums // 2):
        # 位数
        n1 = random.randint(3, 5)
        n2 = random.randint(3, 5)

        # 最高位
        x = random.randint(1, 9)
        y = random.randint(1, 9)

        for i in range(n1 - 1):
            x = x * 10 + random.randint(0, 9)
        for i in range(n2 - 1):
            y = y * 10 + random.randint(0, 9)

        x = x if random.randint(0, 1) else -x
        y = y if random.randint(0, 1) else -y

        train_data.append(f"BOS {x} + {y} = EOS")
        train_data.append(f"BOS {x} - {y} = EOS")
        train_gt.append(f"BOS {x + y} EOS")
        train_gt.append(f"BOS {x - y} EOS")

    random.seed(41)
    for _ in range(test_data_nums // 2):
        # 位数
        n1 = random.randint(3, 5)
        n2 = random.randint(3, 5)

        # 最高位
        x = random.randint(1, 9)
        y = random.randint(1, 9)

        for i in range(n1 - 1):
            x = x * 10 + random.randint(0, 9)
        for i in range(n2 - 1):
            y = y * 10 + random.randint(0, 9)

        x = x if random.randint(0, 1) else -x
        y = y if random.randint(0, 1) else -y

        test_data.append(f"BOS {x} + {y} = EOS")
        test_data.append(f"BOS {x} - {y} = EOS")
        test_gt.append(f"BOS {x + y} EOS")
        test_gt.append(f"BOS {x - y} EOS")
    return train_data, train_gt, test_data, test_gt


class Tokenizer_Digitwise:
    def __init__(self):
        # 固定词表， 只会出现这几种 tokens
        tokens = ["PAD", "BOS", "EOS", "+", "-", "=",
                  "0", "1", "2", "3", "4",
                  "5", "6", "7", "8", "9"]
        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.inv_vocab = {i: token for token, i in self.vocab.items()}
        self.pad_id = self.vocab["PAD"]

    def text2tokens(self, text:str) -> list[str]:
        """
        eg text = "BOS -123 + 456 = EOS"
        return ["BOS", "-", "1", "2", "3", "+", "4", "5", "6", "=", "EOS"]
        """
        parts = text.strip().split()
        tokens = []
        for part in parts:
            if part in ["BOS", "EOS", "+", "-", "="]:
                tokens.append(part)
            elif part.lstrip('-').isdigit():
                tokens.extend(list(part))
            else:
                raise ValueError(f"Unexpected token: {part}")
        return tokens

    def text2ids(self, text:str) -> list[int]:
        tokens = self.text2tokens(text)
        return [self.vocab[token] for token in tokens]

    def pad_sequences(self, seqs):
        max_len = max(len(s) for s in seqs)

        padded = []
        for s in seqs:
            padded.append(s + [self.pad_id] * (max_len - len(s)))

        return padded

    def ids2text(self, ids: list[int]) -> str:
        tokens = [self.inv_vocab[i] for i in ids]
        # 合并数字（同前）
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i] == '-' and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                num_str = '-'
                j = i + 1
                while j < len(tokens) and tokens[j].isdigit():
                    num_str += tokens[j]
                    j += 1
                result.append(num_str)
                i = j
            elif tokens[i].isdigit():
                num_str = ''
                j = i
                while j < len(tokens) and tokens[j].isdigit():
                    num_str += tokens[j]
                    j += 1
                result.append(num_str)
                i = j
            else:
                result.append(tokens[i])
                i += 1
        return " ".join(result)


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, device=device)

    def forward(self, text_ids):
        """
        ids: the ids referred to the tokens
        """
        word_vec = self.embedding(text_ids)
        return word_vec


def evaluate(model, dataloader, device, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            enc_in = x
            dec_in = y[:, :-1]
            target = y[:, 1:]

            logits = model(enc_in, dec_in)

            B, L, V = logits.shape

            loss = criterion(
                logits.reshape(-1, V),
                target.reshape(-1)
            )

            total_loss += loss.item()

            pred = logits.argmax(dim=-1)

            # 忽略 padding
            mask = target != 0

            correct += (pred[mask] == target[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100 * correct / total

    model.train()  # 恢复训练模式

    return avg_loss, acc


"""
test codes
"""
# tokenizer = Tokenizer_Digitwise()
# wordemb = WordEmbedding(len(tokenizer.vocab), 8)
# text = "BOS -123 + 456 = EOS"
# tokens = tokenizer.text2tokens(text)
# ids = tokenizer.text2ids(text)
#
# print (tokenizer.vocab)
# print (tokens)
# print (ids)
#
# word_vec = wordemb(ids)
# print (word_vec)
#
# text2 = tokenizer.ids2text(ids)
# print (text2)



