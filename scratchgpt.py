import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


## ScratchGPT

### Hyperparameters  
""" Divided by 2: 20 min, 6.6it/s, 1.5986 loss, 1.7746 val loss """ 
""" Polytechnique SSH: 23 min, 5.33it/s, 1.0992 train loss, 1.4863 val loss """

#vocab_size = 106        # Size of the vocabulary
batch_size = 64          # Number of batches to train on
block_size = 256         # Context Lenght to look back on in each block
n_embd = 384             # Embedding dimension
n_head = 6               # Number of attention heads (every head is 64 dimension (n_embd/ n_head))
n_layers = 6             # Number of layers for the transformer
max_iter = 2000          # Number of iterations to train
learning_rate = 3e-4     # Learning rate
e = 250                  # Print loss every e iterations
loss_iter = 200          # Average loss estimation over loss_iter iterations
max_new_tokens = 5000    # Number of tokens to generate
dp = 0.2                 # Dropout rate

""" 10 680 576 parameters (cf GPT for formula calculous)"""



TRAIN = False                               # Set to False to load the weights, set to True to train the model
loading_weights = "weights_sp.pth"  # Path to upload or load the weights for model
output_name = "output_sp4.txt"      # Path to upload the output file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}\n')




'''
Load the dataset
'''
# Load the text file
with open('media/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create a vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder and decoder functions
def encode(text):
    return [stoi[c] for c in text]

def decode(encoded_text):
    return ''.join([itos[i] for i in encoded_text])



'''
Encode the text
'''
data = torch.tensor(encode(text), dtype=torch.long)
n = len(data)
train_data = data[:int(0.9 * n)]
val_data = data[int(0.9 * n):]




'''
DataLoader
'''
def get_batch(split):
    """
    Get a batch of data for training or validation
    split: 'train' or 'val'
    """

    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)






'''
Self Attention Head
'''
class SelfAttentionHead(nn.Module):
    """
    Self attention head, for tojens to look at previous for context
    n_embd: embedding size
    head_size: size of the head
    block_size: context length
    mask: mask to prevent future tokens from being attended to

    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dp)

    def forward(self, x):
        """
        Forward pass of the self attention head
        x: input sequence of shape (B, T, C)
        """
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores (scaled attention)
        scores = q @ k.transpose(-2, -1) / (C ** 0.5)                        # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))   # Mask future tokens, only context from previous, one directional attention
        scores = F.softmax(scores, dim=-1)                                   # (B, T, T)
        scores = self.dropout(scores)                                        # Apply dropout to attention scores

        # Compute attention output
        out = scores @ v           # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        
        return out



'''
Multi head attention
'''
class MultiHeadAttention(nn.Module):
    """
    Multi head attention, for tokens to look at previous for context
    """
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
        self.project = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dp)

    def forward(self, x):
        """
        Forward pass of the multi head attention
        x: input sequence of shape (B, T, C)
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, n_heads * head_size = n_embd)
        out = self.dropout(self.project(out))
        return out





'''
Feed forward network
'''
class FeedForward(nn.Module):
    """
    Feed forward network, to think after tokens looked at previous context
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expansion layer from paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # projection layer going back into residuals
            nn.Dropout(dp)                   # dropout for regularization, before residual connection goes back into main network
        )

    def forward(self, x):
        """
        Forward pass of the feed forward network
        x: input sequence of shape (B, T, C)
        """
        return self.net(x)




'''
Block
'''
class Block(nn.Module):
    """
    Transformers Block as a whole, packaging of previous processes
    n_embd: embedding size
    n_head: number of heads
    head_size: size of the head
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, head_size=n_embd // n_head)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass of the block
        x: input sequence of shape (B, T, C)
        """
        x = x + self.attention(self.ln1(x))       ## add residual connection for stability (fork off and add), and layer norm to normalize
        x = x + self.feed_forward(self.ln2(x))    ## add residual connection for stability (fork off and add), and layer norm to normalize
        return x





'''
Model definition
'''
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        
        ## Embedding layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)              ## we encode our vocab into n_embd dimensions lookup table to read logits
        ## Positional encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)           ## we encode our positions into n_embd dimensions
        ## Self attention blocks (4 heads + layer norm)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        ## Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)                                             ## layer norm to normalize the output
        ## Output layer, Linear layer to project the output to vocab size
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        """
        Forward pass of the model: we get scores for next characters of sequence given the input for each vocab
        idx: input sequence of shape (B, T)
        targets: target sequence of shape (B, T)
        """

        B, T = idx.shape
       
        token_emb = self.token_embedding_table(idx)       # (batch_size (B), block_size (T), n_embd (C))
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)      # (T, n_embd)

        ### encode token identity AND position
        x = token_emb + pos_emb                           # (B, T, n_embd)
        ### apply self attention
        x = self.blocks(x)                                # (B, T, n_embd)
        ### apply final layer norm
        x = self.ln_f(x)                                  # (B, T, n_embd)
        ### get logits
        logits = self.lm_head(x)                          # (B, T, vocab_size)


        ## Loss to know how good logits generated are close to targets
        if targets is None:
            loss = None
        
        else:
            ### Reshape logits and targets to compute loss if targets are provided
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            ### Compute loss cross entropy
            loss = F.cross_entropy(logits, targets)

        return logits, loss



    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given the input sequence idx, for every B dimension (batch) over time (T) for max_new_tokens
        idx: input sequence of shape (B, T) to append tokens to
        max_new_tokens: number of tokens to generate
        """

        for _ in tqdm(range(max_new_tokens), desc='Generating tokens'):
            ### Ensure idx compatible with block_size positional encoding
            idx_cond = idx[:, -block_size:]  # (B, T) -> (B, block_size)

            ### get the logits for the next character
            logits, _ = self(idx_cond)       # logits shape: (B, T, C)

            ### Focus on the last time step
            logits = logits[:, -1, :]   # logits shape: (B, C)

            ### Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # probs shape: (B, C)

            ### Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # idx_next shape: (B, 1) (1 pred for each batch)

            ### Append the new token to the input sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx






def estimate_loss(model):
    """
    Estimate the loss on the validation
    Set by averaging over 500 iterations
    Return a dictionary with the loss for train and val
    """
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(loss_iter)
            for k in range(loss_iter):
                xb, yb = get_batch(split)
                logits, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out



'''
Training or loading the model
'''
def train_or_load(model, optimizer, TRAIN, loading_weights):
    """
    Train or load the model
    """
    if TRAIN:
        print('Training the model...')
        model.train()

        loss_suivi = {'iter':[], 'train': [], 'val': []}
        for iter in tqdm(range(max_iter)):

            ## Get the batch
            xb, yb = get_batch('train')

            ## Forward pass
            logits, loss = model(xb, yb)

            ## Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ## Print loss every e iterations
            if iter % e == 0 or iter == max_iter - 1:
                losses = estimate_loss(model)
                print(f'Iter {iter}/{max_iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
                loss_suivi['iter'].append(iter)
                loss_suivi['train'].append(losses['train'])
                loss_suivi['val'].append(losses['val'])
        
        ## Save the model
        print('Saving the model...')
        torch.save(model.state_dict(), "weights/"+loading_weights)
        print(f'Saved model to weights/{loading_weights}')

        ## Save the loss
        print('Saving the loss...')
        df = pd.DataFrame(loss_suivi)
        df.to_csv('loss.csv', index=False)
        print('Saved loss to loss.csv')

    else:
        print('Loading the model...')
        model.load_state_dict(torch.load("weights/"+loading_weights, map_location=device))
        print('Model loaded')



'''
Initialize the model and optimizer
'''
model_blm = BigramLM().to(device)
optimizer = torch.optim.AdamW(model_blm.parameters(), lr=learning_rate)
train_or_load(model_blm, optimizer, TRAIN, loading_weights)


'''
Generate text
'''
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model_blm.generate(context, max_new_tokens)[0].tolist()
print(decode(generated))


'''
Save into a file'
'''
with open("outputs/"+output_name, 'w', encoding='utf-8') as f:
    f.write(decode(generated))
print(f'Saved generated text to outputs/{output_name}')