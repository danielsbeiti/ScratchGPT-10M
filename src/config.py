import torch
import os

# Model hyperparameters
batch_size = 64          # Number of batches to train on
block_size = 256         # Context Length to look back on in each block
n_embd = 384             # Embedding dimension
n_head = 6               # Number of attention heads (each head is 64 dimension (n_embd / n_head))
n_layers = 6             # Number of layers for the transformer
max_iter = 4000          # Number of iterations to train
learning_rate = 3e-4     # Learning rate
e = 250                  # Print loss every e iterations
loss_iter = 200          # Average loss estimation over loss_iter iterations
dp = 0.3                 # Dropout rate

# BPE hyperparameters
num_merges = 2000        # Number of BPE merges to perform
base_vocab_size = 256     # Base vocabulary size (UTF-8 bytes)
vocab_size = base_vocab_size + num_merges  # Total vocabulary size

# Generation hyperparameter
max_new_tokens = 2000    # Number of tokens to generate - CHANGE AS YOU WISH

# Get project root directory (parent directory of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
data_path = os.path.join(PROJECT_ROOT, 'media', 'poem.txt')
weights_dir = os.path.join(PROJECT_ROOT, 'weights/')
outputs_dir = os.path.join(PROJECT_ROOT, 'outputs/')
merges_dir = os.path.join(PROJECT_ROOT, 'merges/')
# Use JSON as primary format for merges
merges_path = os.path.join(merges_dir, 'merges.json')
# The TXT version will be derived from this in the code

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
