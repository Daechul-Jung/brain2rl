import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import load_eeg_data, load_fmri_data, create_dataloader

class CNNFeatureExtractor(nn.Module):
    """
    CNN for feature extraction from brain signals
    """
    def __init__(self, input_channels, input_length, embedding_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Calculate output size after convolutions
        self.output_length = input_length // 8  # After 3 pooling layers with stride 2
    
    def forward(self, x):
        # Input shape: (batch_size, channels, length)
        x = self.conv_layers(x)
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, embedding_dim, max_length=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, embedding_dim)
        return x + self.pe[:, :x.size(1)]

class BrainTokenizer(nn.Module):
    """
    Transformer-based model for brain signal tokenization
    Architecture:
    1. CNN feature extractor to get embeddings from raw signals
    2. Transformer encoder to model temporal dependencies
    3. Output layer to map to token space
    """
    def __init__(self, input_channels, input_length, n_tokens=512, embedding_dim=128, 
                 nhead=8, num_encoder_layers=6, dropout=0.1):
        super(BrainTokenizer, self).__init__()
        
        # Feature extraction
        self.feature_extractor = CNNFeatureExtractor(input_channels, input_length, embedding_dim)
        output_length = self.feature_extractor.output_length
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length=output_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, 
                                                   dim_feedforward=embedding_dim*4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Output projection
        self.token_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, n_tokens)
        )
        
        self.output_length = output_length
        
    def forward(self, x):
        # Input shape: (batch_size, channels, length)
        features = self.feature_extractor(x)  # (batch_size, embedding_dim, output_length)
        
        # Transpose for transformer: (batch_size, output_length, embedding_dim)
        features = features.transpose(1, 2)
        
        # Add positional encoding
        features = self.positional_encoding(features)
        
        # Transformer encoder
        encoded = self.transformer_encoder(features)
        
        # Token prediction
        tokens = self.token_predictor(encoded)
        
        return tokens

def train_tokenizer(args):
    """Train the brain signal tokenizer"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data
    if args.data == 'eeg':
        X, y = load_eeg_data(os.path.join('data', 'eeg'))
        model = BrainTokenizer(input_channels=X.shape[1], input_length=X.shape[2])
    else:  # fmri
        X, y = load_fmri_data(os.path.join('data', 'fmri'))
        model = BrainTokenizer(input_channels=X.shape[1], input_length=X.shape[2])
    
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloader(X, y, batch_size=args.batch_size)
    
    # For tokenization, we'll use a reconstruction loss as a simplified approach
    # In a real-world scenario, you'd want to define a more sophisticated training objective
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', f'tokenization_{args.data}'))
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the model
            tokens = model(inputs)
            
            # For demonstration, we'll use a reconstruction loss
            # The tokens should be decodable back to the input signal
            # This is a simplification - in practice, you'd define a more suitable objective
            batch_size, seq_len, n_tokens = tokens.shape
            target = torch.zeros(batch_size, seq_len, n_tokens, device=device)
            # Set target for each token position based on the ground truth
            for i in range(batch_size):
                for j in range(seq_len):
                    # Simulate a target token distribution
                    target[i, j, :] = torch.softmax(torch.randn(n_tokens, device=device), dim=0)
            
            loss = criterion(tokens, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                
                tokens = model(inputs)
                
                batch_size, seq_len, n_tokens = tokens.shape
                target = torch.zeros(batch_size, seq_len, n_tokens, device=device)
                for i in range(batch_size):
                    for j in range(seq_len):
                        target[i, j, :] = torch.softmax(torch.randn(n_tokens, device=device), dim=0)
                
                loss = criterion(tokens, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join('models', 'tokenization', f'best_{args.data}_tokenizer.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    writer.close()

def evaluate_tokenizer(args):
    """Evaluate the brain signal tokenizer"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data
    if args.data == 'eeg':
        X, y = load_eeg_data(os.path.join('data', 'eeg'))
        model = BrainTokenizer(input_channels=X.shape[1], input_length=X.shape[2])
    else:  # fmri
        X, y = load_fmri_data(os.path.join('data', 'fmri'))
        model = BrainTokenizer(input_channels=X.shape[1], input_length=X.shape[2])
    
    # Load the trained model
    model_path = os.path.join('models', 'tokenization', f'best_{args.data}_tokenizer.pth')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Create dataloaders (we only need the test set)
    _, _, test_loader = create_dataloader(X, y, batch_size=args.batch_size)
    
    # Evaluation
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            tokens = model(inputs)
            
            batch_size, seq_len, n_tokens = tokens.shape
            target = torch.zeros(batch_size, seq_len, n_tokens, device=device)
            for i in range(batch_size):
                for j in range(seq_len):
                    target[i, j, :] = torch.softmax(torch.randn(n_tokens, device=device), dim=0)
            
            loss = criterion(tokens, target)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualization for a few examples
    print("Generating token visualizations for a few examples...")
    for inputs, labels in list(test_loader)[:3]:
        inputs = inputs.to(device)
        tokens = model(inputs)
        
        # Here you might want to visualize the tokens, e.g., with a heatmap
        print(f"Input shape: {inputs.shape}, Token shape: {tokens.shape}")
        print(f"Sample tokens for class {labels[0].item()}:")
        sample_tokens = tokens[0, :5, :5].cpu().numpy()
        print(sample_tokens) 