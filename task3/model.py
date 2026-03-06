import torch
from torch import nn
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, masked=False, device=device):
        super().__init__()

        self.num_heads = num_heads  # H
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # D
        self.masked = masked

        assert hidden_dim%num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(input_dim, hidden_dim, device=device)
        self.k_linear = nn.Linear(input_dim, hidden_dim, device=device)
        self.v_linear = nn.Linear(input_dim, hidden_dim, device=device)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, device=device)
    def forward(self, q_in, k_in, v_in):
        """
        self_attn时， q_in = k_in = v_in = (B, L, D)
        cross_attn时, q_in = (B, T, D), k_in = v_in = (B, L, D)
                      q_in是 decoder_in 经过selfatten后的结果，
                      k_in, v_in 是 encoder_out的结果
        """
        B, T, D = q_in.shape
        L = k_in.shape[1]

        # q = (B, L, H)
        q = self.q_linear(q_in)
        k = self.k_linear(k_in)
        v = self.v_linear(v_in)

        # q = (B, L, num_head, head_dim) -> (B, num_head, T, head_dim)
        # k = (B, L, num_head, head_dim) -> (B, num_head, head_dim, L) 为了方便点积 @
        # v = (B, L, num_head, head_dim) -> (B, num_head, L, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0,2,3,1).contiguous()
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()

        # attn_score = (B, num_head, T, L)
        attn_score = q @ k / math.sqrt(self.head_dim)

        if self.masked:
            mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn_score = attn_score.masked_fill(mask.bool(), -1e9)

        # attn_score = (B, num_head, T, L)
        # (B, num_head, T, L) @ (B, num_heads, L, H) = (B, N, T, H)
        attn_score = torch.softmax(attn_score, dim=-1) @ v

        # 还原多头 (B, T, H*N)
        return self.out_proj(attn_score.permute(0, 2, 1, 3).contiguous().view(B, T, self.hidden_dim))

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-9, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # 可学习参数：gamma (scale), beta (bias)
            self.gamma = nn.Parameter(torch.ones(emb_dim)).to(device)
            self.beta = nn.Parameter(torch.zeros(emb_dim)).to(device)

    def forward(self, x):
        # x: (B, L, D)
        mean = x.mean(dim=-1, keepdim=True)  # (B, L, 1)
        var = x.var(dim=-1, keepdim=True)  # (B, L, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim_FFN=None):
        super().__init__()
        if hidden_dim_FFN==None:
            hidden_dim_FFN=input_dim
        self.FF = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_FFN),
            nn.ReLU(),
            nn.Linear(hidden_dim_FFN, input_dim),
        )

    def forward(self, x):
        return self.FF(x)

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim_attn, num_heads=4, dropout=0.1, hidden_dim_FFN=None):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, hidden_dim_attn, num_heads, masked=False)
        self.LayerNorm1 = LayerNorm(input_dim)
        if hidden_dim_FFN == None:
            hidden_dim_FFN = 2 * input_dim
        self.FFN = FFN(input_dim, hidden_dim_FFN)
        self.LayerNorm2 = LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.LayerNorm1(x)
        x = self.dropout(self.self_attn(x, x, x)) + res
        res = x
        x = self.LayerNorm2(x)
        return self.dropout(self.FFN(x)) + res


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_blocks=4, dropout=0.1):
        super().__init__()
        self.inp_dim = input_dim
        self.num_heads = num_heads

        self.encoder_blocks = nn.ModuleList([EncoderBlock(input_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
                              for _ in range(num_blocks)])
    def forward(self,x):
        """
        x = (B, L, D)
        """
        for block in self.encoder_blocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.masked_selfattn = MultiheadAttention(input_dim, hidden_dim, num_heads, masked=True)
        self.layer_norm1 = LayerNorm(input_dim)

        self.crossatten = MultiheadAttention(input_dim, hidden_dim, num_heads)
        self.layer_norm2 = LayerNorm(input_dim)

        self.FFN = FFN(input_dim, None)
        self.layer_norm3 = LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_out, dec_in):
        res = dec_in
        dec_in = self.layer_norm1(dec_in)
        out = self.dropout(self.masked_selfattn(dec_in, dec_in, dec_in)) + res

        res = out
        out = self.layer_norm2(out)
        out = self.dropout(self.crossatten(out, prev_out, prev_out)) + res

        res = out
        out = self.layer_norm3(out)
        return self.dropout(self.FFN(out)) + res

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_blocks=4, dropout=0.1):
        super().__init__()

        self.inp_dim = input_dim
        self.num_heads = num_heads

        self.decoder_blocks = nn.ModuleList([DecoderBlock(input_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
                                             for _ in range(num_blocks)])

    def forward(self, prev_out, dec_in):
        """
        x = (B, L, D)
        """
        for block in self.decoder_blocks:
            dec_in = block(prev_out, dec_in)
        return dec_in


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=16):
        super().__init__()
        # pe = (max_len, input_dim)
        # position = (max_len, 1)
        # div_term = (input_dim // 2, 1)
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim)
        )

        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe = (max_len, input_dim) -> (1, max_len, input_dim)
        pe = pe.unsqueeze(0)

        # 不参与训练
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x = (B, L, D)
        """
        L = x.shape[1]
        return x + self.pe[:, :L, :]


from utils import WordEmbedding
class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_blocks,
                vocab_size, max_len=16, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_heads, num_blocks, dropout)
        self.decoder = Decoder(input_dim, hidden_dim, num_heads, num_blocks, dropout)
        self.embedding = WordEmbedding(vocab_size, input_dim)
        self.pe = PositionalEncoding(input_dim, max_len)
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, src_ids, target_ids):
        src_ids = src_ids.to(device)
        target_ids = torch.tensor(target_ids).to(device)
        scr = self.pe(self.embedding(src_ids))
        tgt = self.pe(self.embedding(target_ids))

        enc_out = self.encoder(scr)
        dec_out = self.decoder(enc_out, tgt)
        logits = self.fc(dec_out)

        return logits



