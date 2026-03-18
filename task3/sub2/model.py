import torch
import torch.nn as nn
"""
Decoder-Only Model, 仅做next-token-prediction
"""
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, block_size=96):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout = 0.15,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape

        # token embedding
        tok_emb = self.token_emb(x)

        # position embedding
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos)

        x = tok_emb + pos_emb

        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask)

        x = self.ln(x)
        logits = self.head(x)

        return logits
    
def generate(model, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()

    for _ in range(max_new_tokens):
        # 只取最后 block_size
        input_cond = input_ids[:, -model.block_size:]

        # 前向
        logits = model(input_cond)  # [B, T, vocab]
        logits = logits[:, -1, :]   # 只取最后一个 token

        # temperature
        logits = logits / temperature

        # top-k
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)

        # 采样
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        # 拼接
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids



def evaluate(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            # 多 GPU 平均
            loss_value = accelerator.gather(loss.detach()).mean().item()
            total_loss += loss_value

    return total_loss / len(val_loader)
