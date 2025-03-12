import torch
import os
import config
from bpe import train_bpe, encode_with_bpe, decode_with_bpe

def load_text(file_path):
    """Load text data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def prepare_data(file_path):
    """
    Prepare data for training and validation
    Returns:
        train_data: Training data tensor
        val_data: Validation data tensor
        vocab_size: Size of vocabulary
        vocab: Vocabulary mapping
        merges: BPE merges
    """
    # Load text
    text = load_text(file_path)
    
    # Create merges directory if it doesn't exist
    os.makedirs(os.path.dirname(config.merges_path), exist_ok=True)
    
    # Train BPE
    print(f"Training BPE on {file_path}")
    vocab, merges, encoded = train_bpe(text, config.num_merges, save_path=config.merges_path)
    
    # Encode the text
    # encoded = encode_with_bpe(text, merges)
    data = torch.tensor(encoded, dtype=torch.long)
    
    # Split into train and validation
    n = len(data)
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):]
    
    return train_data, val_data, config.vocab_size, vocab, merges

def get_batch(split, train_data, val_data):
    """Get a batch of data for training or validation"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)
