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

class SensorActionDataset(Dataset):
    """Custom dataset for the sensor-based action classification"""
    def __init__(self, data, labels, window_size=100, overlap=0.5, transform=None):
        """
        Args:
            data: Sensor data with shape (n_samples, n_channels)
            labels: Action labels (0-4 for 5 different actions)
            window_size: Window size for segmentation
            overlap: Overlap between consecutive windows (0-1)
            transform: Optional transform to be applied on a sample
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.transform = transform
        
        self.segments = []
        self.segment_labels = []
        self._create_segments()
        
    def _create_segments(self):
        """Create segments with sliding windows"""
        n_samples = len(self.data)
        
        for start in range(0, n_samples - self.window_size, self.stride):
            end = start + self.window_size
            
            segment = self.data[start:end]
            # Use the most common label in the window
            window_labels = self.labels[start:end]
            label = np.bincount(window_labels).argmax()
            
            self.segments.append(segment)
            self.segment_labels.append(label)
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.segment_labels[idx]
        
        if self.transform:
            segment = self.transform(segment)
        
        # Convert to tensor
        segment = torch.tensor(segment, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        # Transpose the segment to have shape [channels, time_steps]
        segment = segment.transpose(0, 1)
        
        return segment, label

def load_sensor_data(data_dir, subject_ids=None):
    """
    Load data from the sensor dataset
    
    Args:
        data_dir: Directory containing the data files
        subject_ids: List of subject IDs to load
    
    Returns:
        data_dict: Dictionary with subject as keys, containing data and labels
    """
    data_dict = {}
    
    for subject_id in subject_ids:
        # Construct file paths
        data_file = os.path.join(data_dir, f'{subject_id}.csv')
        
        # Check if file exists
        if not os.path.exists(data_file):
            print(f"File for subject {subject_id} not found. Skipping...")
            continue
        
        # Load data
        print(f"Loading subject {subject_id}...")
        df = pd.read_csv(data_file)
        
        # Extract features and labels
        X = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        y = df['label'].values
        
        data_dict[subject_id] = {
            'data': X,
            'labels': y,
            'data_df': df
        }
            
    return data_dict

def preprocess_sensor_data(data):
    """
    Preprocess sensor data
    Args:
        data: Sensor data with shape (n_samples, n_channels)
    Returns:
        Preprocessed data
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled

class SensorActionClassifier(nn.Module):
    """
    CNN model for sensor-based action classification
    """
    def __init__(self, n_channels=6, n_times=100, n_classes=5, dropout_rate=0.3):
        super(SensorActionClassifier, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # Feature extraction
        self.feature_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # Additional feature extraction
        self.feature_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate size after convolutions and pooling
        time_after_pool = n_times // 8  # Three pooling layers with stride 2
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * time_after_pool, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, channels, times)
        x = self.temporal_conv(x)
        x = self.feature_conv(x)
        x = self.feature_conv2(x)
        x = self.fc(x)
        return x  # No sigmoid for multi-class classification

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
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Test loss and accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return test_loss, accuracy

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    data_dir = 'data'
    subject_ids = ['SUBJ_000206', 'SUBJ_001430']  # Example subject IDs
    
    data_dict = load_sensor_data(data_dir, subject_ids)
    
    # Combine data from all subjects
    all_data = []
    all_labels = []
    
    for subject_id in data_dict:
        data = data_dict[subject_id]['data']
        labels = data_dict[subject_id]['labels']
        
        # Preprocess data
        data = preprocess_sensor_data(data)
        
        all_data.append(data)
        all_labels.append(labels)
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = SensorActionDataset(X_train, y_train)
    val_dataset = SensorActionDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = SensorActionClassifier()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    main() 