import os
import json
from tqdm import tqdm



def get_stats(ids):
    """
    Count frequency of adjacent token pairs
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
       counts[pair] = counts.get(pair, 0) + 1
    return counts



def merge_pair(tokens, pair, idx):
    """
    Merge the most frequent pair of tokens in the list.
    Args:
        tokens (list): List of tokens.
        pair (tuple): Pair of tokens to merge.
        idx (int): Index of the pair to merge.
    Returns:
        list: List of tokens with the pair merged.
    """
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens




def tokenize(text):
    """
    Tokenize the text into a list of tokens.
    Args:
        text (str): Text to tokenize.
    Returns:
        list: List of tokens.
    """
    return list(map(int, text.encode('utf-8')))



def bpe_train(tokens, num_merges):
    """
    Apply BPE to the list of tokens.
    Args:
        tokens (list): List of tokens.
        num_merges (int): Number of merges to apply.
    Returns:
        list: List of tokens after applying BPE.
    """
    print(f'Training BPE with {num_merges} merges...')
    merges = {}
    for i in tqdm(range(num_merges)):
        stats = get_stats(tokens)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        tokens = merge_pair(tokens, pair, idx)
        merges[pair] = idx
    return tokens, merges





def create_vocab_from_merges(merges):
    """
    Create vocabulary from merges
    Args:
        merges (dict): Dictionary of merges
    Returns:
        dict: Vocabulary mapping token IDs to byte sequences
    """
    # Start with base UTF-8 byte vocabulary
    vocab = {idx: bytes([idx]) for idx in range(256)}
    
    # Add merged tokens to vocabulary
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        
    return vocab





def save_merges(merges, vocab, filepath):
    """
    Save merges to a file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('#version: 0.2\n')
        f.write('#\n')
        f.write('# merges\n')
        f.write('#\n')

        for (p0, p1), idx in merges.items():
            f.write(f'[{vocab[p0].decode("utf-8", errors="replace")}] + [{vocab[p1].decode("utf-8", errors="replace")}] -> [{(vocab[idx]).decode("utf-8", errors="replace")}] as {idx}\n')


def save_merges_json(merges, filepath):
    """
    Save merges dictionary to a JSON file
    Args:
        merges (dict): Dictionary of merges with tuple keys
        filepath (str): Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert tuple keys to strings since JSON can't have tuple keys
    serializable_merges = {f"{p0},{p1}": idx for (p0, p1), idx in merges.items()}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_merges, f, indent=2)







def train_bpe(text, num_merges, save_path=None):
    """
    Train BPE on text and return vocabulary and merges
    """
    tokens = tokenize(text)
    tokens, merges = bpe_train(tokens, num_merges)
    vocab = create_vocab_from_merges(merges)
    
    if save_path:
        # Get paths for JSON and TXT files
        json_path = save_path
        txt_path = save_path.replace('.json', '.txt')
            
        # Save JSON version for direct loading (only the merges dictionary)
        save_merges_json(merges, json_path)
        
        # Save TXT version for human inspection
        save_merges(merges, vocab, txt_path)
    
    return vocab, merges, tokens




def encode_with_bpe(text, merges):
    """
    Encode text using BPE
    Args:
        text (str): Text to encode
        merges (dict): Dictionary of merges
    Returns:
        list: List of encoded tokens
    """
    tokens = tokenize(text)

    # Apply merges while possible
    while len(tokens) > 1:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda x: merges.get(x, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge_pair(tokens, pair, idx)
    
    return tokens

def decode_with_bpe(encoded_text, vocab):
    """
    Decode BPE encoded tokens
    Args:
        encoded_text (list): List of encoded tokens
        vocab (dict): Vocabulary mapping token IDs to byte sequences
    Returns:
        str: Decoded text
    """
    return b"".join([vocab[i] for i in encoded_text]).decode('utf-8', errors='replace')
