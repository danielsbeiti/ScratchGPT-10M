import torch
import os
import argparse
import config
from data import prepare_data
from model import BigramLM
from trainer import train_model, load_model
from generate import generate_text, save_generated_text



def main(train=False, weights_file="model.pth", output_file="output.txt"):
    """
    Main function to either train the model or generate text
    Args:
        train (bool): If True, train the model. If False, generate text.
        weights_file (str): Filename for saving/loading model weights.
        output_file (str): Filename for saving generated text.
    """
    print(f"Using device: {config.device}")
    
    # Ensure directories exist
    os.makedirs(config.weights_dir, exist_ok=True)
    os.makedirs(config.outputs_dir, exist_ok=True)
    os.makedirs(config.merges_dir, exist_ok=True)

    # Prepare data
    train_data, val_data, vocab_size, vocab, merges = prepare_data(config.data_path)
    print(f"Vocabulary size: {vocab_size}")
    
    if train:
        print(f"BPE merges will be saved in both JSON format (for loading) and TXT format (for visualization)")
        print(f"  - JSON: {config.merges_path}")
        print(f"  - TXT: {config.merges_path.replace('.json', '.txt')}\n\n")

    # Initialize model
    model = BigramLM(vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    weights_path = os.path.join(config.weights_dir, weights_file)
    output_path = os.path.join(config.outputs_dir, output_file)
    

    # Train or load model
    if train:
        model = train_model(model, optimizer, train_data, val_data, weights_path)
    else:
        model = load_model(model, weights_path)
    

    # Generate text
    print("Generating text...")
    generated_text = generate_text(model, vocab)
    print("\nSample of generated text:")
    print(generated_text[:500] + "...\n")
    
    # Save generated text
    save_generated_text(generated_text, output_path)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ScratchGPT: A custom GPT implementation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--weights", type=str, default="model.pth", help="Weights filename")
    parser.add_argument("--output", type=str, default="output.txt", help="Output filename")
    
    args = parser.parse_args()
    
    main(
        train=args.train,
        weights_file=args.weights,
        output_file=args.output
    )