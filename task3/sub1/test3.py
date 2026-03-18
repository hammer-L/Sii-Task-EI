import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = [
'0','1','2','3','4','5','6','7','8','9',
'+',
'<PAD>',
'<BOS>',
'<EOS>'
]

def generate_dataset(
        n_samples=20000,
        max_digit=5,
        test_ratio=0.2,
        seed=42):

    random.seed(seed)

    digit_choices = [3,4,5]

    train_x = []
    train_y = []

    xs = []
    ys = []

    for _ in range(n_samples):

        d1 = random.choice(digit_choices)
        d2 = random.choice(digit_choices)

        a = random.randint(10**(d1-1), 10**d1 - 1)
        b = random.randint(10**(d2-1), 10**d2 - 1)

        # encoder 输入 padding
        a_str = str(a).zfill(max_digit)
        b_str = str(b).zfill(max_digit)

        encoder_input = f"{a_str}+{b_str}"

        result = str(a+b)

        decoder_input = "<BOS>" + result
        decoder_target = result + "<EOS>"

        xs.append(encoder_input)
        ys.append((decoder_input, decoder_target))

    train_x, test_x, train_y, test_y = train_test_split(
        xs, ys, test_size=test_ratio, random_state=seed
    )

    return train_x, train_y, test_x, test_y


class PositionalEncoding(nn.Module):

    def __init__(self,d_model,max_len=50):
        super().__init__()

        pe=torch.zeros(max_len,d_model)

        pos=torch.arange(0,max_len).unsqueeze(1)

        div=torch.exp(torch.arange(0,d_model,2)*(-torch.log(torch.tensor(10000.0))/d_model))

        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)

        self.pe=pe.unsqueeze(0)

    def forward(self,x):

        return x+self.pe[:,:x.size(1)].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,d_model)
        self.pos = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.pos(self.embed(src))
        tgt=self.pos(self.embed(tgt))

        # tgt是decoder输入，为了防作弊，生成mask
        tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        out=self.transformer(src,tgt,tgt_mask=tgt_mask)
        return self.fc(out)


class DecoderModel(nn.Module):

    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=4
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.pos(self.embed(x))
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x, mask)
        return self.fc(out)
