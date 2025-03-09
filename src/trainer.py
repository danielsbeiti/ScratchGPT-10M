import torch
import pandas as pd
from tqdm import tqdm
import os
import config
from data import get_batch

def estimate_loss(model, train_data, val_data):
    """Estimate loss on train and validation data"""
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(config.loss_iter)
            for k in range(config.loss_iter):
                xb, yb = get_batch(split, train_data, val_data)
                logits, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out

def train_model(model, optimizer, train_data, val_data, weights_path):
    """Train the model"""
    model.train()
    
    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    loss_tracking = {'iter': [], 'train': [], 'val': []}
    
    for iter in tqdm(range(config.max_iter), desc="Training"):
        # Get batch
        xb, yb = get_batch('train', train_data, val_data)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Print loss every e iterations
        if iter % config.e == 0 or iter == config.max_iter - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f'Iter {iter}/{config.max_iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
            loss_tracking['iter'].append(iter)
            loss_tracking['train'].append(losses['train'])
            loss_tracking['val'].append(losses['val'])
    
    # Save the model
    print('Saving the model...')
    torch.save(model.state_dict(), weights_path)
    print(f'Saved model to {weights_path}')
    
    # Save the loss data
    print('Saving the loss data...')
    df = pd.DataFrame(loss_tracking)
    df.to_csv('loss.csv', index=False)
    print('Saved loss to loss.csv')
    
    return model

def load_model(model, weights_path):
    """Load a trained model"""
    print(f'Loading model from {weights_path}...')
    model.load_state_dict(torch.load(weights_path, map_location=config.device))
    print('Model loaded successfully')
    return model
