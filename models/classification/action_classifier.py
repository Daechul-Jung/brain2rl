import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import load_eeg_data, load_fmri_data, create_dataloader

class EEGConvNet(nn.Module):
    """
    CNN model for EEG signal classification
    Architecture:
    1. Temporal convolutions to extract time-frequency features
    2. Spatial convolutions to model relationships between channels
    3. Fully connected layers for classification
    """
    def __init__(self, n_channels=32, n_times=512, n_classes=6, dropout_rate=0.5):
        super(EEGConvNet, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), stride=1, padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(n_channels, 1), stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        )
        
        # Calculate the size after convolutions and pooling
        time_after_pool = n_times // 16  # Two pooling layers with stride 4
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * time_after_pool, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, channels, times)
        # Add dimension for conv2d which expects (batch_size, filters, channels, times)
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.fc(x)
        return x

def train_classifier(args):
    """Train the action classification model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load data
    if args.data == 'eeg':
        X, y = load_eeg_data(os.path.join('data', 'eeg'))
        model = EEGConvNet(n_channels=X.shape[1], n_times=X.shape[2], n_classes=len(np.unique(y)))
    else:  # fmri
        X, y = load_fmri_data(os.path.join('data', 'fmri'))
        
    
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloader(X, y, batch_size=args.batch_size)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', f'classification_{args.data}'))
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join('models', 'classification', f'best_{args.data}_classifier.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    writer.close()

def evaluate_classifier(args):
    """Evaluate the action classification model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data
    if args.data == 'eeg':
        X, y = load_eeg_data(os.path.join('data', 'eeg'))
        model = EEGConvNet(n_channels=X.shape[1], n_times=X.shape[2], n_classes=len(np.unique(y)))
    else:  # fmri
        X, y = load_fmri_data(os.path.join('data', 'fmri'))
        
    
    # Load the trained model
    model_path = os.path.join('models', 'classification', f'best_{args.data}_classifier.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Create dataloaders (we only need the test set)
    _, _, test_loader = create_dataloader(X, y, batch_size=args.batch_size)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%") 