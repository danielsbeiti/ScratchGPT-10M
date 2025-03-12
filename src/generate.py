import torch
import os
import config
from bpe import decode_with_bpe




def generate_text(model, vocab, max_new_tokens=None):
    """
    Generate text using the model
    Args:
        model: The trained model
        vocab: Vocabulary for decoding
        max_new_tokens: Maximum number of tokens to generate
    Returns:
        Generated text
    """

    if max_new_tokens is None:
        max_new_tokens = config.max_new_tokens
    
    # Start with an empty context
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    
    # Generate new tokens
    generated = model.generate(context, max_new_tokens, vocab)[0].tolist()
    
    # Decode the generated tokens
    text = decode_with_bpe(generated, vocab)
    
    return text




def save_generated_text(text, output_path):
    """
    Save generated text to a file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Saved generated text to {output_path}')