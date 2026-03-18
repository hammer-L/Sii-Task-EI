# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

accelerator = Accelerator()
name = "wikitext-2"

# 1. 加载数据
dataset = load_dataset("wikitext", f"{name}-raw-v1")
val_texts = dataset["validation"]["text"]
train_texts = dataset["train"]["text"]
# print ("texts[1]", texts[1])

# 拼接成一个长文本, 这样模型能学习跨句子依赖
train_text = "\n".join(train_texts)
val_text = "\n".join(val_texts)
# 2. tokenizer
# 使用 GPT2 tokenizer（byte-level BPE）
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT2 没有 pad_token，手动指定
tokenizer.pad_token = tokenizer.eos_token
# print("tokenizer.eos_token", tokenizer.eos_token)

# tokenize 整个文本
# return_tensors="pt" → 直接转成 PyTorch tensor
train_ids = tokenizer(train_text, return_tensors="pt")["input_ids"][0]
val_ids = tokenizer(val_text, return_tensors="pt")["input_ids"][0]


# In[ ]:


class GPTDataset(Dataset):
    """
    把一维 token 序列切成 (x, y) 对
    x: 当前序列
    y: 向右偏移1位（next token）
    """

    def __init__(self, input_ids, block_size=128):
        self.input_ids = input_ids
        self.block_size = block_size

    def __len__(self):
        # 每个样本长度是 block_size
        return len(self.input_ids) - self.block_size

    def __getitem__(self, idx):
        # x: 当前窗口
        x = self.input_ids[idx:idx + self.block_size]

        # y: 右移一位（预测下一个 token）
        y = self.input_ids[idx + 1:idx + self.block_size + 1]

        return x, y


# In[ ]:


block_size = 96   # 上下文长度
batch_size = 16

# dataset
train_dataset = GPTDataset(train_ids, block_size)
val_dataset = GPTDataset(val_ids, block_size)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset,batch_size=batch_size,
    sampler=train_sampler,drop_last=True)

val_loader = DataLoader(val_dataset,batch_size=batch_size,
    sampler=val_sampler,drop_last=False)

# for x, y in train_loader:
#     print("x shape:", x.shape)  # [B, T]
#     print("y shape:", y.shape)  # [B, T]
#     break


# In[ ]:

from model import MiniGPT, generate
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
import os
import shutil


log_dir=f"runs/gpt_{name}_exp"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # 删除旧日志，避免曲线混淆
writer = SummaryWriter(log_dir)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = accelerator.device

# 超参数
epochs = 2
lr = 1e-4
warmup_ratio = 0.05

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 用scheduler 变化学习率， 先warmup， 再余弦decay
num_training_steps = len(train_loader) * epochs
num_warmup_steps = int(warmup_ratio * num_training_steps)  # 5% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 用accelerate 
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler)

# 采样试试
def sample_text(model, tokenizer, prompt="The meaning of life is"):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


# evaluate函数
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

            loss_value = accelerator.gather(loss.detach()).mean().item()
            total_loss += loss_value

    # 同步所有 GPU
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        answer = sample_text(unwrapped_model, tokenizer)
    else:
        answer = None

    return total_loss / len(val_loader), answer


global_step = 0
last_val_loss = 100
global_ans = []
print("start training")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, disable=not accelerator.is_main_process)

    for step, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)  # [B, T, vocab]

        # reshape 做 cross entropy
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        # 多 gpu, 全局平均
        loss_value = accelerator.gather(loss.detach()).mean().item()
        total_loss += loss_value

        # 显示loss
        if (step + 1) % 500 == 0:
            progress_bar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        if (global_step + 1) % 2500 == 0:
            accelerator.wait_for_everyone()
            val_loss, answer = evaluate(model, val_loader)
            global_ans.append(answer)
            if accelerator.is_main_process:
                writer.add_scalar("val/loss", val_loss, global_step)
                print (f"{global_step:}{answer}")

                # 更好就保存参数
                if last_val_loss >= val_loss and global_step >= 10000:
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save({
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step
                    }, f"runs/gpt_wiki2_epoch_{epoch+1}.pt")
                last_val_loss = val_loss


        # tensorboard记录
        if accelerator.is_main_process:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr",scheduler.get_last_lr()[0],global_step)
        global_step += 1

    # 等一下gpu
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Epoch {epoch} avg loss: {total_loss / len(train_loader):.4f}")

print (global_ans)
writer.close()


