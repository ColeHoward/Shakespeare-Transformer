import torch
from models import NGramLanguageModel


# define hyperparameters
batch_size = 32
block_size = 256  # context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
# device = 'cuda' if torch.cuda.is_available() else "cpu"
n_embedding_dimensions = 384
num_heads = 6
num_layers = 6
dropout = .2
# ----------------------------------------

torch.manual_seed(2023)
# read data from tiny shakespeare
with open('data/input.txt') as f:
    text = f.read()

# all the unique characters that are in this text
chars = sorted(set(text))
vocab_size = len(chars)
# create mapping from characters to integers
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # takes string, outputs list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # takes list of integers, outputs strings

# create train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(.9 * len(text))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idxs])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idxs])
    # x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()  # change model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = NGramLanguageModel(vocab_size, n_embedding_dimensions, block_size, num_heads, num_layers, dropout)
# TRAIN MODEL ------------------------------------------------------
# model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step: {i}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample batch from trainset
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# SAMPLE FROM LEARNED DISTRIBUTION --------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long)  # , device=device)
out = model.generate(context, new_max_tokens=1000)
gen_text = decode(out[0].tolist())
print(gen_text)
