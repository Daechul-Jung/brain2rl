import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.tokenization.brain_tokenizer import BrainTokenizer
from models.classification.eeg_action_classification import GraspAndLiftEEGDataset, load_grasp_and_lift_data, preprocess_eeg, EEGActionClassifier
from models.rl.brain_rl import BrainGuidedAgent

class EEGTokenizationPipeline:
    """
    Pipeline for tokenizing EEG signals and preparing them for RL
    """
    def __init__(self, data_dir, tokenizer_path=None, classifier_path=None, 
                 window_size=500, overlap=0.5, device=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load the tokenizer
        self.load_tokenizer(tokenizer_path)
        
        # Load the classifier
        self.load_classifier(classifier_path)
        
    def load_tokenizer(self, tokenizer_path):
        """Load the brain tokenizer model"""
        # We'll determine input dimensions when we load data
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        
    def load_classifier(self, classifier_path):
        """Load the EEG action classifier model"""
        # We'll determine input dimensions when we load data
        self.classifier = None
        self.classifier_path = classifier_path
        
    def load_data(self, subject_ids=None, series_ids=None):
        """Load and preprocess the EEG data"""
        print("Loading EEG data...")
        self.data_dict = load_grasp_and_lift_data(self.data_dir, subject_ids, series_ids)
        
        # Process data for all subjects/series
        self.processed_data = []
        self.processed_events = []
        
        for subject_id in self.data_dict:
            for series_id in self.data_dict[subject_id]:
                # Get data and events
                data = self.data_dict[subject_id][series_id]['data']
                events = self.data_dict[subject_id][series_id]['events']
                
                # Preprocess the data
                data = preprocess_eeg(data)
                
                self.processed_data.append(data)
                self.processed_events.append(events)
        
        # Concatenate data and events if we have multiple subjects/series
        if len(self.processed_data) > 0:
            self.processed_data = np.vstack(self.processed_data)
            self.processed_events = np.vstack(self.processed_events)
            
            # Initialize models with correct dimensions if not done yet
            if self.tokenizer is None:
                n_channels = self.processed_data.shape[1]
                input_length = self.window_size  # We'll use the window size for tokenizer
                
                print(f"Initializing tokenizer with {n_channels} channels and input length {input_length}")
                self.tokenizer = BrainTokenizer(input_channels=n_channels, input_length=input_length)
                
                if self.tokenizer_path and os.path.exists(self.tokenizer_path):
                    self.tokenizer.load_state_dict(torch.load(self.tokenizer_path, map_location=self.device))
                    print(f"Loaded tokenizer from {self.tokenizer_path}")
                else:
                    print(f"No tokenizer weights loaded. Using untrained tokenizer.")
                
                self.tokenizer = self.tokenizer.to(self.device)
                self.tokenizer.eval()
            
            if self.classifier is None and self.classifier_path:
                n_channels = self.processed_data.shape[1]
                n_classes = self.processed_events.shape[1]
                
                print(f"Initializing classifier with {n_channels} channels and {n_classes} classes")
                self.classifier = EEGActionClassifier(n_channels=n_channels, n_times=self.window_size, n_classes=n_classes)
                
                if os.path.exists(self.classifier_path):
                    self.classifier.load_state_dict(torch.load(self.classifier_path, map_location=self.device))
                    print(f"Loaded classifier from {self.classifier_path}")
                else:
                    print(f"Classifier path {self.classifier_path} not found.")
                
                self.classifier = self.classifier.to(self.device)
                self.classifier.eval()
        
        # Create dataset with sliding windows
        self.dataset = GraspAndLiftEEGDataset(
            self.processed_data, 
            self.processed_events, 
            window_size=self.window_size, 
            overlap=self.overlap
        )
        
        print(f"Created dataset with {len(self.dataset)} windows")
        return self.dataset
    
    def tokenize_batch(self, batch_data):
        """Tokenize a batch of EEG data"""
        with torch.no_grad():
            # Ensure data is on the correct device
            batch_tensor = batch_data.to(self.device)
            
            # Generate tokens
            tokens = self.tokenizer(batch_tensor)
            
            return tokens
    
    def classify_batch(self, batch_data):
        """Classify a batch of EEG data to identify actions"""
        if self.classifier is None:
            return None
            
        with torch.no_grad():
            # Ensure data is on the correct device
            batch_tensor = batch_data.to(self.device)
            
            # Generate action predictions
            outputs = self.classifier(batch_tensor)
            
            # Convert to binary predictions
            predictions = (outputs > 0.5).float()
            
            return predictions
    
    def create_dataloader(self, batch_size=32, shuffle=True):
        """Create a dataloader for the dataset"""
        return DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
    
    def run_pipeline(self, batch_size=32, max_batches=None):
        """
        Run the complete pipeline: 
        1. Load data
        2. Create batches
        3. Tokenize each batch
        4. Classify actions (if classifier is available)
        5. Return tokens and actions
        """
        dataloader = self.create_dataloader(batch_size=batch_size)
        
        all_tokens = []
        all_actions = []
        all_labels = []
        
        # Process batches
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Processing batches")):
            if max_batches is not None and i >= max_batches:
                break
                
            # Transpose inputs to have shape [batch_size, channels, time_steps]
            inputs = inputs.transpose(1, 2)
            
            # Tokenize the batch
            tokens = self.tokenize_batch(inputs)
            
            # Classify the batch (if classifier is available)
            if self.classifier is not None:
                actions = self.classify_batch(inputs)
            else:
                actions = labels  # Use ground truth labels if no classifier
            
            # Store results
            all_tokens.append(tokens.cpu().numpy())
            all_actions.append(actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions.numpy())
            all_labels.append(labels.numpy())
        
        # Concatenate results
        all_tokens = np.vstack(all_tokens)
        all_actions = np.vstack(all_actions)
        all_labels = np.vstack(all_labels)
        
        return all_tokens, all_actions, all_labels
    
    def visualize_token_action_relationship(self, tokens, actions, n_samples=5):
        """Visualize the relationship between tokens and actions"""
        if n_samples > tokens.shape[0]:
            n_samples = tokens.shape[0]
        
        action_names = [
            "HandStart", 
            "FirstDigitTouch", 
            "BothStartLoadPhase", 
            "LiftOff", 
            "Replace", 
            "BothReleased"
        ]
        
        for i in range(n_samples):
            plt.figure(figsize=(15, 8))
            
            # Plot token heatmap
            plt.subplot(2, 1, 1)
            plt.imshow(tokens[i, :, :100], aspect='auto', cmap='viridis')
            plt.colorbar(label='Token Value')
            plt.title(f'Sample {i+1}: Brain Tokens')
            plt.xlabel('Token Dimension (first 100)')
            plt.ylabel('Token Sequence')
            
            # Plot action probabilities
            plt.subplot(2, 1, 2)
            actions_i = actions[i]
            plt.bar(range(len(actions_i)), actions_i)
            plt.xticks(range(len(actions_i)), action_names, rotation=45)
            plt.ylim([0, 1])
            plt.title('Action Probabilities')
            
            plt.tight_layout()
            plt.savefig(f'token_action_sample_{i+1}.png')
            plt.close()
        
        print(f"Visualizations saved for {n_samples} samples")
    
    def prepare_for_rl(self, tokens, actions):
        """
        Prepare tokens and actions for reinforcement learning.
        Returns a dataset that can be used by the BrainGuidedAgent.
        """
        # For simplicity, we'll just return the tokens and actions
        # In a real application, you might want to create a more complex dataset
        return tokens, actions


class EEGActionVisualization:
    """
    Utilities for visualizing EEG data and corresponding actions
    """
    @staticmethod
    def plot_eeg_with_actions(eeg_data, actions, window_size=500, title=None):
        """
        Plot EEG data with action indicators
        
        Args:
            eeg_data: EEG data with shape (n_samples, n_channels)
            actions: Action data with shape (n_samples, n_classes)
            window_size: Window size for plotting
            title: Optional title for the plot
        """
        n_samples, n_channels = eeg_data.shape
        n_classes = actions.shape[1]
        
        action_names = [
            "HandStart", 
            "FirstDigitTouch", 
            "BothStartLoadPhase", 
            "LiftOff", 
            "Replace", 
            "BothReleased"
        ]
        
        plt.figure(figsize=(15, 10))
        
        # Plot EEG data (first 8 channels)
        ax1 = plt.subplot(2, 1, 1)
        for i in range(min(8, n_channels)):
            plt.plot(eeg_data[:window_size, i], label=f'Channel {i+1}')
        plt.legend()
        plt.title(title or 'EEG Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        # Plot actions
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        for i in range(n_classes):
            plt.plot(actions[:window_size, i], label=action_names[i])
        plt.legend()
        plt.title('Actions')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.ylim([-0.1, 1.1])
        
        plt.tight_layout()
        plt.savefig('eeg_actions_visualization.png')
        plt.close()
        
        print("EEG and action visualization saved to 'eeg_actions_visualization.png'")
    
    @staticmethod
    def create_action_heatmap(actions, title=None):
        """
        Create a heatmap of actions over time
        
        Args:
            actions: Action data with shape (n_samples, n_classes)
            title: Optional title for the plot
        """
        action_names = [
            "HandStart", 
            "FirstDigitTouch", 
            "BothStartLoadPhase", 
            "LiftOff", 
            "Replace", 
            "BothReleased"
        ]
        
        plt.figure(figsize=(12, 6))
        plt.imshow(actions.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Action Probability')
        plt.title(title or 'Action Heatmap')
        plt.xlabel('Time')
        plt.ylabel('Action')
        plt.yticks(range(len(action_names)), action_names)
        
        plt.tight_layout()
        plt.savefig('action_heatmap.png')
        plt.close()
        
        print("Action heatmap saved to 'action_heatmap.png'")


def main():
    # Example usage
    data_dir = os.path.join('data', 'train')
    tokenizer_path = os.path.join('models', 'tokenization', 'best_eeg_tokenizer.pth')
    classifier_path = 'best_eeg_action_classifier.pth'
    
    # Create the pipeline
    pipeline = EEGTokenizationPipeline(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        classifier_path=classifier_path
    )
    
    # Load data (use first 2 subjects for demonstration)
    subject_ids = [1, 2]
    pipeline.load_data(subject_ids=subject_ids)
    
    # Run the pipeline
    tokens, actions, labels = pipeline.run_pipeline(batch_size=16, max_batches=10)
    
    print(f"Generated {tokens.shape[0]} token sequences")
    print(f"Token shape: {tokens.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Visualize token-action relationship
    pipeline.visualize_token_action_relationship(tokens, actions, n_samples=3)
    
    # Visualize EEG data with actions
    sample_idx = 0
    sample_data = pipeline.dataset.segments[sample_idx]
    sample_actions = pipeline.dataset.labels[sample_idx]
    
    EEGActionVisualization.plot_eeg_with_actions(
        sample_data, 
        sample_actions.reshape(1, -1).repeat(len(sample_data), axis=0)
    )
    
    # Create action heatmap for multiple windows
    action_data = np.vstack([pipeline.dataset.labels[i] for i in range(min(100, len(pipeline.dataset)))])
    EEGActionVisualization.create_action_heatmap(action_data)
    
    # Prepare for RL (optional)
    rl_tokens, rl_actions = pipeline.prepare_for_rl(tokens, actions)
    
    print("Pipeline completed successfully!")
    
    # Connect to RL (optional demonstration)
    if False:  # Set to True to demonstrate RL integration
        env_name = "Pendulum-v0"  # A simple continuous control environment
        agent = BrainGuidedAgent(
            env_name=env_name,
            tokenizer_path=tokenizer_path,
            tokenizer_type='eeg'
        )
        
        # Example of using tokens in RL
        print("Running RL agent with tokenized brain signals...")
        agent.evaluate(num_episodes=2)


if __name__ == "__main__":
    main() 