import torch
import os

# Model hyperparameters
batch_size = 64          # Number of batches to train on
block_size = 256         # Context Length to look back on in each block
n_embd = 384             # Embedding dimension
n_head = 6               # Number of attention heads (each head is 64 dimension (n_embd / n_head))
n_layers = 6             # Number of layers for the transformer
max_iter = 2000          # Number of iterations to train
learning_rate = 3e-4     # Learning rate
e = 250                  # Print loss every e iterations
loss_iter = 200          # Average loss estimation over loss_iter iterations
dp = 0.2                 # Dropout rate

# Generation hyperparameter
max_new_tokens = 5000    # Number of tokens to generate - CHANGE AS YOU WISH

# Get project root directory (parent directory of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
data_path = os.path.join(PROJECT_ROOT, 'media', 'shakespeare.txt')
weights_dir =os.path.join(PROJECT_ROOT, 'weights/')
outputs_dir = os.path.join(PROJECT_ROOT, 'outputs/')

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
