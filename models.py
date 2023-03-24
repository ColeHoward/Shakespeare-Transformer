import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # since tril isn't noramlly an nn.Module parameter, you have to register it like so:
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C ** -.5  # (B T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        v = self.value(x)  # (B, T, C)
        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        # self, head_size, n_embd, block_size, dropout
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # add residual connection
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """simple linear layer followed by a non-linearity"""
    def __init__(self, n_embedding_dimensions, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding_dimensions, 4*n_embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4*n_embedding_dimensions, n_embedding_dimensions),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transform block: token "communication" followed by computation"""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # add residual connections
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x


class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embedding_dimensions, block_size, num_heads, num_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding_dimensions)
        # encode the position of tokens, not just the value
        self.position_embedding_table = nn.Embedding(block_size, n_embedding_dimensions)
        self.blocks = nn.Sequential(*[Block(n_embedding_dimensions, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(n_embedding_dimensions)
        self.lm_head = nn.Linear(n_embedding_dimensions, vocab_size)  # language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B, T, n_emb)
        positional_embedding = self.position_embedding_table(torch.arange(T))  # (T, C)
        x = token_embeddings + positional_embedding
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # (B, T, Vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, new_max_tokens):
        # idx is (B, T) array of indicies that are in the current context
        for _ in range(new_max_tokens):
            # crop index to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time-step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
