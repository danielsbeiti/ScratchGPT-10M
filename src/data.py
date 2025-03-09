import torch
import config


def load_text(file_path):
    """
    Load text data from file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text



def create_vocab(text):
    """
    Create vocabulary from text
    Args:
        text (str): Input text
    Returns:
        characters list,
        vocabulary size, 
        character to integer mapping, 
        integer to character mapping
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return chars, vocab_size, stoi, itos



def encode(text, stoi):
    """
    Encode text to integers
    Args:
        text (str): Input text
        stoi (dict): Character to integer mapping
    Returns:
        list: List of integers representing the text
    """
    return [stoi[c] for c in text]

def decode(encoded_text, itos):
    """
    Decode integers to text
    Args:
        encoded_text (list): List of integers
        itos (dict): Integer to character mapping
    Returns:
        str: Decoded text
    """
    return ''.join([itos[i] for i in encoded_text])






def prepare_data(file_path):
    """Prepare data for training and validation"""

    text = load_text(file_path)
    chars, vocab_size, stoi, itos = create_vocab(text)
    
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = len(data)
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):]
    
    return train_data, val_data, vocab_size, stoi, itos



def get_batch(split, train_data, val_data):
    """Get a batch of data for training or validation"""

    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)
