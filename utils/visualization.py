import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_eeg_signal(eeg_data, sample_idx=0, channels=None, title=None, figsize=(15, 8)):
    """
    Plot EEG signal channels over time
    Args:
        eeg_data: EEG data with shape (samples, channels, time)
        sample_idx: Index of the sample to plot
        channels: List of channel indices to plot (None for all)
        title: Plot title
        figsize: Figure size
    """
    if channels is None:
        channels = range(min(5, eeg_data.shape[1]))  # Default to first 5 channels
    
    plt.figure(figsize=figsize)
    
    for i in channels:
        plt.plot(eeg_data[sample_idx, i], label=f'Channel {i+1}')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'EEG Signal - Sample {sample_idx+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

def plot_fmri_slice(fmri_data, sample_idx=0, slice_idx=None, title=None, figsize=(12, 10)):
    """
    Plot a slice of fMRI data
    Args:
        fmri_data: fMRI data with shape (samples, voxels, time)
        sample_idx: Index of the sample to plot
        slice_idx: Index of the time slice to plot (None for middle)
        title: Plot title
        figsize: Figure size
    """
    if slice_idx is None:
        slice_idx = fmri_data.shape[2] // 2  # Default to middle slice
    
    # Reshape the voxels into a 3D volume (this is a simplified approximation)
    voxel_dim = int(np.cbrt(fmri_data.shape[1]))
    volume = fmri_data[sample_idx, :voxel_dim**3, slice_idx].reshape(voxel_dim, voxel_dim, voxel_dim)
    
    # Plot middle slices along each axis
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    axs[0].imshow(volume[voxel_dim//2, :, :], cmap='viridis')
    axs[0].set_title('Sagittal View')
    axs[0].axis('off')
    
    axs[1].imshow(volume[:, voxel_dim//2, :], cmap='viridis')
    axs[1].set_title('Coronal View')
    axs[1].axis('off')
    
    axs[2].imshow(volume[:, :, voxel_dim//2], cmap='viridis')
    axs[2].set_title('Axial View')
    axs[2].axis('off')
    
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'fMRI Data - Sample {sample_idx+1}, Time {slice_idx+1}')
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), title='Confusion Matrix'):
    """
    Plot confusion matrix
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def plot_brain_tokens(tokens, max_tokens=10, max_dim=10, figsize=(12, 10), title='Brain Tokens'):
    """
    Visualize brain tokens as a heatmap
    Args:
        tokens: Token tensor with shape (batch_size, seq_len, token_dim)
        max_tokens: Maximum number of tokens to display
        max_dim: Maximum dimensions to display
        figsize: Figure size
        title: Plot title
    """
    # Convert to numpy if PyTorch tensor
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    
    # Get the first sample
    token_subset = tokens[0, :max_tokens, :max_dim]
    
    plt.figure(figsize=figsize)
    
    # Create a colormap
    cmap = LinearSegmentedColormap.from_list('brain_cmap', ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000'])
    
    # Plot heatmap
    plt.imshow(token_subset, cmap=cmap, aspect='auto')
    plt.colorbar(label='Token Value')
    plt.title(title)
    plt.xlabel('Token Dimension')
    plt.ylabel('Sequence Position')
    
    # Add tick marks
    plt.xticks(range(max_dim), [f'Dim {i+1}' for i in range(max_dim)])
    plt.yticks(range(max_tokens), [f'Pos {i+1}' for i in range(max_tokens)])
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_training_history(train_losses, val_losses, train_accs=None, val_accs=None, figsize=(15, 6), title='Training History'):
    """
    Plot training history
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies (optional)
        val_accs: List of validation accuracies (optional)
        figsize: Figure size
        title: Plot title
    """
    epochs = range(1, len(train_losses) + 1)
    
    if train_accs is not None and val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle(title)
    else:
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        fig = plt.gcf()
    
    plt.tight_layout()
    
    return fig

def visualize_agent_performance(rewards, window_size=10, figsize=(12, 6), title='Agent Performance'):
    """
    Visualize agent performance over episodes
    Args:
        rewards: List of episode rewards
        window_size: Window size for moving average
        figsize: Figure size
        title: Plot title
    """
    # Calculate moving average
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=figsize)
    plt.plot(rewards, 'b-', alpha=0.3, label='Episode Reward')
    plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'r-', label=f'{window_size}-Episode Moving Average')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

def save_figure(fig, filename, directory='figures', dpi=300):
    """
    Save figure to file
    Args:
        fig: Matplotlib figure
        filename: Output filename
        directory: Output directory
        dpi: DPI for saving
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save figure
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filepath}") 