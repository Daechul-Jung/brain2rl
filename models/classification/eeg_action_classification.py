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
import glob



class GraspAndLiftEEGDataset(Dataset):
    """Custom dataset for the Grasp and Lift EEG dataset"""
    def __init__(self, data, events, window_size=500, overlap=0.5, transform=None):
        """
        Args:
            data: EEG data with shape (n_samples, n_channels)
            events: Event data with shape (n_samples, n_classes)
            window_size: Window size for segmentation
            overlap: Overlap between consecutive windows (0-1)
            transform: Optional transform to be applied on a sample
        """
        self.data = data
        self.events = events
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.transform = transform
        
        self.segments = []
        self.labels = []
        self._create_segments()
        
    def _create_segments(self):
        """Create segments with sliding windows"""
        n_samples = len(self.data)
        
        for start in range(0, n_samples - self.window_size, self.stride):
            end = start + self.window_size
            
            segment = self.data[start:end]
            
            window_labels = self.events[start:end].mean(axis=0)
            window_labels = (window_labels > 0.5).astype(np.float32)
            
            self.segments.append(segment)
            self.labels.append(window_labels)
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        
        if self.transform:
            segment = self.transform(segment)
        
        # Convert to tensor
        segment = torch.tensor(segment, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        # Transpose the segment to have shape [channels, time_steps]
        segment = segment.transpose(0, 1)
        
        return segment, label

def load_grasp_and_lift_data(data_dir, subject_ids=None, series_ids=None):
    """
    Load data from the Grasp and Lift EEG Dataset
    
    Args:
        data_dir: Directory containing the data files
        subject_ids: List of subject IDs to load (1-12)
        series_ids: List of series IDs to load (1-8)
    
    Returns:
        data_dict: Dictionary with subject and series as keys, containing data and events
    """
    if subject_ids is None:
        subject_ids = range(1, 13)  # All subjects
    if series_ids is None:
        series_ids = range(1, 9)    # All series
    
    data_dict = {}
    
    for subject_id in subject_ids:
        data_dict[f'subj{subject_id}'] = {}
        
        for series_id in series_ids:
            # Construct file paths
            data_file = os.path.join(data_dir, f'subj{subject_id}_series{series_id}_data.csv')
            events_file = os.path.join(data_dir, f'subj{subject_id}_series{series_id}_events.csv')
            
            # Check if files exist
            if not os.path.exists(data_file) or not os.path.exists(events_file):
                print(f"Files for subject {subject_id}, series {series_id} not found. Skipping...")
                continue
            
            # Load data
            print(f"Loading subject {subject_id}, series {series_id}...")
            data_df = pd.read_csv(data_file)
            events_df = pd.read_csv(events_file)
            
            # Extract features and labels
            # Assuming first column is ID
            X = data_df.iloc[:, 1:].values  # EEG channels (exclude ID column)
            y = events_df.iloc[:, 1:].values  # Events (exclude ID column)
            
            data_dict[f'subj{subject_id}'][f'series{series_id}'] = {
                'data': X,
                'events': y,
                'data_df': data_df,
                'events_df': events_df
            }
            
    return data_dict

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

class EEGActionClassifier(nn.Module):
    """
    CNN model for EEG-based action classification
    """
    def __init__(self, n_channels=32, n_times=500, n_classes=6, dropout_rate=0.5):
        super(EEGActionClassifier, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # Feature extraction
        self.feature_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate size after convolutions and pooling
        time_after_pool = n_times // 4  # Two pooling layers with stride 2
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * time_after_pool, 256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, channels, times)
        x = self.temporal_conv(x)
        x = self.feature_conv(x)
        x = self.fc(x)
        return torch.sigmoid(x)  # For multi-label classification

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

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_dir = os.path.join('data', 'train')
    
    # For demonstration, use first 3 subjects only
    subject_ids = [1, 2, 3]
    data_dict = load_grasp_and_lift_data(data_dir, subject_ids=subject_ids)
    
    # Prepare dataset for training
    all_data = []
    all_events = []
    
    for subject_id in data_dict:
        for series_id in data_dict[subject_id]:
            # Get data and events
            data = data_dict[subject_id][series_id]['data']
            events = data_dict[subject_id][series_id]['events']
            
            # Preprocess data
            data = preprocess_eeg(data)
            
            all_data.append(data)
            all_events.append(events)
    
    # Concatenate data and events
    all_data = np.vstack(all_data)
    all_events = np.vstack(all_events)
    
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, all_events, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create datasets
    window_size = 500  # ~2 seconds of data
    train_dataset = GraspAndLiftEEGDataset(X_train, y_train, window_size=window_size)
    val_dataset = GraspAndLiftEEGDataset(X_val, y_val, window_size=window_size)
    test_dataset = GraspAndLiftEEGDataset(X_test, y_test, window_size=window_size)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get the number of channels and classes
    n_channels = X_train.shape[1]
    n_classes = y_train.shape[1]
    
    # Create model
    model = EEGActionClassifier(n_channels=n_channels, n_times=window_size, n_classes=n_classes)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, 
        n_epochs=30, patience=5
    )
    
    # Evaluate model
    test_loss, test_acc, class_metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Print class-wise metrics
    print("\nClass-wise metrics:")
    for class_idx, metrics in class_metrics.items():
        print(f"{class_idx}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training history plot saved to 'training_history.png'")

if __name__ == "__main__":
    main() 