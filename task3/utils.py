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

    random.seed(99)
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


def predict(model, src_tokens, tgt_tokens, tokenizer, max_len=16, device=device):
    """
    model: 训练好的 Transformer
    src_tokens: DataLoader 产出的题目 Tensor (B, L)
    tgt_tokens: DataLoader 产出的答案 Tensor (B, L_tgt)
    tokenizer: Tokenizer_Digitwise 的实例
    """
    model.eval()

    # 从 tokenizer 对象直接获取 ID
    bos_id = tokenizer.vocab['BOS']
    eos_id = tokenizer.vocab['EOS']
    batch_size = src_tokens.size(0)

    with torch.no_grad():
        # 1. 编码器阶段
        src_tokens = src_tokens.to(device)
        src_emb = model.pe(model.embedding(src_tokens))
        enc_out = model.encoder(src_emb)

        # 2. 自回归解码
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device) * bos_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            dec_emb = model.pe(model.embedding(generated))
            logits = model.decoder(enc_out, dec_emb)
            logits = model.fc(logits)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # 如果整批都生成了 EOS，提前结束
            finished |= (next_token.squeeze() == eos_id)
            if finished.all(): break

    # 3. 统计结果
    correct_count = 0
    batch_results = []

    for i in range(batch_size):
        # 提取模型生成的 IDs (去掉 BOS，截取到 EOS)
        p_ids = generated[i].tolist()[1:]
        p_ids = p_ids[:p_ids.index(eos_id)] if eos_id in p_ids else p_ids

        # 提取真值 IDs (去掉 BOS，截取到 EOS)
        t_ids = tgt_tokens[i].tolist()[1:]
        t_ids = t_ids[:t_ids.index(eos_id)] if eos_id in t_ids else t_ids

        # 全匹配检查
        is_correct = (p_ids == t_ids)
        if is_correct:
            correct_count += 1

        # 使用 tokenizer.inv_vocab 还原文本
        # 注意：这里直接用 "".join 拼接数字，或者调用 tokenizer 现有的解码逻辑
        pred_str = "".join([tokenizer.inv_vocab[idx] for idx in p_ids])
        true_str = "".join([tokenizer.inv_vocab[idx] for idx in t_ids])

        batch_results.append({
            "pred": pred_str,
            "true": true_str,
            "status": "✅" if is_correct else "❌"
        })

    acc = correct_count / batch_size
    return acc, batch_results

# 结合 predict函数使用， 对每次 evaluate过程进行封装
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_acc = 0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            acc, batch_results = predict(model, x, y, tokenizer, device=device)
            total_acc += acc
            count += 1

    model.train()
    return total_acc / count, batch_results

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



