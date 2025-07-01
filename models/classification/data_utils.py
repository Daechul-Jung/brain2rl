import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess_eeg(data):
    """
    Preprocess EEG data
    Args:
        data: EEG data with shape (n_samples, n_channels)
    Returns:
        Preprocessed data
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled


def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=50, patience=10):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        n_epochs: Number of epochs
        patience: Early stopping patience
        
    Returns:
        Trained model, training history
    """
    model = model.to(device)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            
            # For multi-label, use threshold of 0.5 for prediction
            preds = (outputs > 0.5).float()
            train_correct += (preds == targets).float().sum().item()
            train_total += targets.numel()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                
                # For multi-label, use threshold of 0.5 for prediction
                preds = (outputs > 0.5).float()
                val_correct += (preds == targets).float().sum().item()
                val_total += targets.numel()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_eeg_action_classifier.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_eeg_action_classifier.pth'))
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on (cpu or cuda)
        
    Returns:
        Test loss, test accuracy, class-wise metrics
    """
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    
    # For multi-label classification
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            test_loss += loss.item() * inputs.size(0)
            
            # For multi-label, use threshold of 0.5 for prediction
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    
    # Concatenate predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate overall accuracy
    test_acc = np.mean((all_preds == all_targets).flatten())
    
    # Calculate class-wise metrics
    class_metrics = {}
    for i in range(all_targets.shape[1]):
        class_preds = all_preds[:, i]
        class_targets = all_targets[:, i]
        
        true_pos = np.sum((class_preds == 1) & (class_targets == 1))
        false_pos = np.sum((class_preds == 1) & (class_targets == 0))
        true_neg = np.sum((class_preds == 0) & (class_targets == 0))
        false_neg = np.sum((class_preds == 0) & (class_targets == 1))
        
        accuracy = (true_pos + true_neg) / len(class_targets)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[f'class_{i}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return test_loss, test_acc, class_metrics