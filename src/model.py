import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import config

class SelfAttentionHead(nn.Module):
    """
    Self attention head, for tokens to look at previous for context
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size)
        self.query = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)

        # Create a mask to prevent looking ahead
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dp)

    def forward(self, x):
        """
        Forward pass of the self attention head
        param x: Input tensor of shape (B: batch_size, T:context_size, C:embedding_dim)
        return: Output tensor of shape (B: batch_size, T:context_size, C:head_size)
        """
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores (scaled dot product)
        scores = q @ k.transpose(-2, -1) / (C ** 0.5)                         # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))    # Apply mask to prevent looking ahead, one directional attention
        scores = F.softmax(scores, dim=-1)                                    # Apply softmax to get attention weights
        scores = self.dropout(scores)                                         # Apply dropout to attention weights

        # Compute attention output
        out = scores @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi head attention"""
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
        self.project = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, x):
        """
        Forward pass of the multi head attention
        param x: Input tensor of shape (B: batch_size, T:context_size, C:embedding_dim)
        return: Output tensor of shape (B: batch_size, T:context_size, C:embedding_dim)
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.project(out))
        return out



class FeedForward(nn.Module):
    """
    Feed forward network, for tokens to look at themselves after attention
    param n_embd: Embedding dimension
    param n_hidden: Hidden dimension
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # 4 times the embedding dimension from paper (expansion layer)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # Projection layer goiong back to residuals (retroactve layer from paper)
            nn.Dropout(config.dp)
        )

    def forward(self, x):
        """
        Forward pass of the feed forward network
        param x: Input of shape (B: batch_size, T:context_size, C:embedding_dim)
        return: Output of shape (B: batch_size, T:context_size, C:embedding_dim)
        """
        return self.net(x)




class Block(nn.Module):
    """
    Transformer Block with multi-head attention and feed forward network
    Packaging of previous defined modules
    param n_embd: Embedding dimension
    param n_head: Number of attention heads
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, head_size=n_embd // n_head)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))         # Add residual connection
        x = x + self.feed_forward(self.ln2(x))      # Add residual connection
        return x





# GPT Language Model final construction

class BigramLM(nn.Module):
    """GPT language model"""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Embedding layers for tokens and positions
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)                # Tokens embedding in n_embd dimensions, lookup table to read logits
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)      # Position encoding in n_embd dimensios
        
        # Transformer blocks (n_heads of self attention and feed forward)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layers)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)                                             # Final layer norm to normalize the output
        self.lm_head = nn.Linear(config.n_embd, vocab_size)                                 # Project output to vocab size



    def forward(self, idx, targets=None):
        """
        Forward pass of the model: get scores for next characters of the input
        param idx: Input of shape (B: batch_size, T:context_size)
        param targets: Target of shape (B: batch_size, T:context_size)
        return: Output of shape (B: batch_size, T:context_size, vocab_size)
        """

        B, T = idx.shape
        
        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=config.device)
        pos_emb = self.position_embedding_table(pos)
        
        ## Combine token and position embeddings
        x = token_emb + pos_emb
        ## Pass through self attention transformer blocks
        x = self.blocks(x)
        ## Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for loss calculation
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Calculate cross entropy loss
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss




    def generate(self, idx, max_new_tokens, vocab):
        """
        Generate new tokens given the input sequence
        param idx: Input of shape (B: batch_size, T:context_size)
        param max_new_tokens: Maximum number of tokens to generate
        param vocab: Vocabulary mapping for decoding
        return: Generated tokens of shape (B: batch_size, T:context_size + max_new_tokens)
        """

        for _ in tqdm(range(max_new_tokens), desc='Generating tokens'):
            # Get only the last block_size tokens
            idx_cond = idx[:, -config.block_size:]    # (B, T) -> (B, block_size)
            
            # Get predictions (logits) for the next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # focus on last step only
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
