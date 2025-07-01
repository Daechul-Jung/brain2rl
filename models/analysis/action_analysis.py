import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from models.classification.eeg_action_classification import load_grasp_and_lift_data, preprocess_eeg, EEGActionClassifier
from models.pipelines.eeg_to_rl_pipeline import EEGTokenizationPipeline, EEGActionVisualization

class ActionAnalyzer:
    """
    Analyze and visualize EEG-based actions from the Grasp and Lift dataset
    """
    def __init__(self, data_dir, classifier_path=None, window_size=500, overlap=0.5, device=None):
        self.data_dir = data_dir
        self.classifier_path = classifier_path
        self.window_size = window_size
        self.overlap = overlap
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Action names for the dataset
        self.action_names = [
            "HandStart", 
            "FirstDigitTouch", 
            "BothStartLoadPhase", 
            "LiftOff", 
            "Replace", 
            "BothReleased"
        ]
        
        # Load classifier if provided
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            print(f"Loading classifier from {classifier_path}")
            # We'll initialize it when we know the data dimensions
        
    def load_data(self, subject_ids=None, series_ids=None):
        """Load data from the Grasp and Lift dataset"""
        print("Loading data...")
        self.data_dict = load_grasp_and_lift_data(self.data_dir, subject_ids, series_ids)
        
        # Process data
        self.subject_data = {}
        for subject_id in self.data_dict:
            self.subject_data[subject_id] = {}
            for series_id in self.data_dict[subject_id]:
                # Get data and events
                data = self.data_dict[subject_id][series_id]['data']
                events = self.data_dict[subject_id][series_id]['events']
                
                # Preprocess data
                data_processed = preprocess_eeg(data)
                
                self.subject_data[subject_id][series_id] = {
                    'data': data_processed,
                    'events': events
                }
        
        # Initialize classifier if provided
        if self.classifier_path and self.classifier is None:
            # Get dimensions from the first subject and series
            first_subject = list(self.subject_data.keys())[0]
            first_series = list(self.subject_data[first_subject].keys())[0]
            n_channels = self.subject_data[first_subject][first_series]['data'].shape[1]
            n_classes = self.subject_data[first_subject][first_series]['events'].shape[1]
            
            self.classifier = EEGActionClassifier(n_channels=n_channels, n_times=self.window_size, n_classes=n_classes)
            self.classifier.load_state_dict(torch.load(self.classifier_path, map_location=self.device))
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()
        
        return self.subject_data
    
    def analyze_action_patterns(self):
        """Analyze action patterns in the dataset"""
        if not hasattr(self, 'subject_data'):
            print("Please load data first using load_data()")
            return
        
        # Analyze action co-occurrence and sequence
        action_counts = {name: 0 for name in self.action_names}
        action_co_occurrence = np.zeros((len(self.action_names), len(self.action_names)))
        action_transitions = np.zeros((len(self.action_names), len(self.action_names)))
        
        for subject_id in self.subject_data:
            for series_id in self.subject_data[subject_id]:
                events = self.subject_data[subject_id][series_id]['events']
                
                # Count action occurrences
                action_sum = events.sum(axis=0)
                for i, count in enumerate(action_sum):
                    action_counts[self.action_names[i]] += count
                
                # Count co-occurrences
                for i in range(len(self.action_names)):
                    for j in range(len(self.action_names)):
                        if i != j:
                            # Count when both actions are active simultaneously
                            co_occur = np.sum((events[:, i] > 0.5) & (events[:, j] > 0.5))
                            action_co_occurrence[i, j] += co_occur
                
                # Analyze transitions
                prev_actions = np.zeros(len(self.action_names))
                for t in range(1, len(events)):
                    curr_actions = (events[t] > 0.5).astype(int)
                    
                    # Check for transitions (0->1)
                    for i in range(len(self.action_names)):
                        if curr_actions[i] == 1 and events[t-1, i] <= 0.5:
                            # This action just started
                            for j in range(len(self.action_names)):
                                if prev_actions[j] == 1:
                                    # Action j was active before action i started
                                    action_transitions[j, i] += 1
                    
                    prev_actions = curr_actions
        
        # Normalize transition matrix
        row_sums = action_transitions.sum(axis=1, keepdims=True)
        action_transitions_norm = action_transitions / (row_sums + 1e-10)
        
        results = {
            'action_counts': action_counts,
            'action_co_occurrence': action_co_occurrence,
            'action_transitions': action_transitions,
            'action_transitions_norm': action_transitions_norm
        }
        
        return results
    
    def plot_action_analysis(self, results):
        """Plot action analysis results"""
        # 1. Action counts
        plt.figure(figsize=(10, 6))
        counts = [results['action_counts'][name] for name in self.action_names]
        plt.bar(self.action_names, counts)
        plt.title('Action Counts in Dataset')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('action_counts.png')
        plt.close()
        
        # 2. Action co-occurrence
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['action_co_occurrence'], 
                   xticklabels=self.action_names, 
                   yticklabels=self.action_names, 
                   annot=True, fmt='g', cmap='viridis')
        plt.title('Action Co-occurrence Matrix')
        plt.tight_layout()
        plt.savefig('action_co_occurrence.png')
        plt.close()
        
        # 3. Action transitions
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['action_transitions_norm'], 
                   xticklabels=self.action_names, 
                   yticklabels=self.action_names, 
                   annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Action Transition Probabilities')
        plt.xlabel('Next Action')
        plt.ylabel('Current Action')
        plt.tight_layout()
        plt.savefig('action_transitions.png')
        plt.close()
        
        print("Analysis plots saved to files.")
    
    def identify_action_sequences(self, min_sequence_length=5):
        """Identify common action sequences in the dataset"""
        if not hasattr(self, 'subject_data'):
            print("Please load data first using load_data()")
            return
        
        all_sequences = []
        
        for subject_id in self.subject_data:
            for series_id in self.subject_data[subject_id]:
                events = self.subject_data[subject_id][series_id]['events']
                
                # Convert to binary indicators
                binary_events = (events > 0.5).astype(int)
                
                # Find sequences of actions
                current_sequence = []
                for t in range(len(binary_events)):
                    active_actions = [self.action_names[i] for i in range(len(self.action_names)) 
                                    if binary_events[t, i] == 1]
                    
                    if active_actions:
                        # Store as a tuple (time_point, active_actions)
                        current_sequence.append((t, frozenset(active_actions)))
                    elif current_sequence:
                        # End of a sequence
                        if len(current_sequence) >= min_sequence_length:
                            all_sequences.append(current_sequence)
                        current_sequence = []
                
                # Add last sequence if not empty
                if current_sequence and len(current_sequence) >= min_sequence_length:
                    all_sequences.append(current_sequence)
        
        # Count unique sequences
        sequence_counts = {}
        for sequence in all_sequences:
            # Convert to a hashable representation: just the actions without time points
            actions_only = tuple([actions for _, actions in sequence])
            if actions_only in sequence_counts:
                sequence_counts[actions_only] += 1
            else:
                sequence_counts[actions_only] = 1
        
        # Sort by frequency
        sorted_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 10 sequences
        top_sequences = sorted_sequences[:10]
        
        return top_sequences, all_sequences
    
    def plot_common_sequences(self, top_sequences):
        """Plot the most common action sequences"""
        fig, axes = plt.subplots(min(5, len(top_sequences)), 1, figsize=(12, 3*min(5, len(top_sequences))))
        
        if len(top_sequences) == 0:
            print("No sequences found.")
            return
        
        if len(top_sequences) == 1:
            axes = [axes]
        
        for i, (sequence, count) in enumerate(top_sequences[:5]):
            # Create a matrix to visualize the sequence
            seq_matrix = np.zeros((len(self.action_names), len(sequence)))
            
            for t, action_set in enumerate(sequence):
                for action in action_set:
                    # Convert from frozenset back to list of actions
                    for action_name in list(action):
                        if action_name in self.action_names:
                            action_idx = self.action_names.index(action_name)
                            seq_matrix[action_idx, t] = 1
            
            # Plot the sequence
            sns.heatmap(seq_matrix, cmap='Blues', 
                      xticklabels=range(len(sequence)), 
                      yticklabels=self.action_names,
                      ax=axes[i], cbar=False)
            axes[i].set_title(f'Sequence {i+1} (Count: {count})')
            axes[i].set_xlabel('Time Step')
            
        plt.tight_layout()
        plt.savefig('common_action_sequences.png')
        plt.close()
        
        print(f"Common action sequences plot saved.")
    
    def evaluate_classifier(self, test_subject_ids=None):
        """Evaluate the classifier on test subjects"""
        if self.classifier is None:
            print("No classifier loaded. Please provide a valid classifier_path.")
            return
        
        if not hasattr(self, 'subject_data'):
            print("Please load data first using load_data()")
            return
        
        if test_subject_ids is None:
            # Use all subjects
            test_subject_ids = list(self.subject_data.keys())
        
        # Prepare test data
        all_preds = []
        all_true = []
        
        for subject_id in test_subject_ids:
            print(f"Evaluating on subject {subject_id}...")
            for series_id in self.subject_data[subject_id]:
                data = self.subject_data[subject_id][series_id]['data']
                events = self.subject_data[subject_id][series_id]['events']
                
                # Process data in windows
                for start in range(0, len(data) - self.window_size, int(self.window_size * (1 - self.overlap))):
                    end = start + self.window_size
                    window = data[start:end]
                    
                    # Get ground truth (majority vote in window)
                    window_events = events[start:end]
                    true_labels = (window_events.mean(axis=0) > 0.5).astype(int)
                    
                    # Make prediction
                    with torch.no_grad():
                        window_tensor = torch.tensor(window, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
                        window_tensor = window_tensor.to(self.device)
                        outputs = self.classifier(window_tensor)
                        preds = (outputs > 0.5).float().cpu().numpy().squeeze()
                    
                    all_preds.append(preds)
                    all_true.append(true_labels)
        
        # Convert to arrays
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        # Calculate metrics
        accuracy = np.mean((all_preds == all_true).all(axis=1))
        
        # Per-class metrics
        class_metrics = {}
        for i, action_name in enumerate(self.action_names):
            true_pos = np.sum((all_preds[:, i] == 1) & (all_true[:, i] == 1))
            false_pos = np.sum((all_preds[:, i] == 1) & (all_true[:, i] == 0))
            true_neg = np.sum((all_preds[:, i] == 0) & (all_true[:, i] == 0))
            false_neg = np.sum((all_preds[:, i] == 0) & (all_true[:, i] == 1))
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[action_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Plot confusion matrices for each action
        plt.figure(figsize=(15, 10))
        for i, action_name in enumerate(self.action_names):
            plt.subplot(2, 3, i+1)
            cm = confusion_matrix(all_true[:, i], all_preds[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title(f'{action_name} (F1: {class_metrics[action_name]["f1"]:.2f})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig('classifier_confusion_matrices.png')
        plt.close()
        
        # Print metrics
        print(f"Overall accuracy: {accuracy:.4f}")
        print("\nPer-class metrics:")
        metrics_df = pd.DataFrame(class_metrics).T
        print(metrics_df)
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'predictions': all_preds,
            'true_labels': all_true
        }
    
    def visualize_subject_actions(self, subject_id, series_id, window_start=0, window_length=1000):
        """Visualize actions for a specific subject and series"""
        if not hasattr(self, 'subject_data'):
            print("Please load data first using load_data()")
            return
        
        if subject_id not in self.subject_data or series_id not in self.subject_data[subject_id]:
            print(f"Subject {subject_id}, series {series_id} not found in loaded data.")
            return
        
        # Get data
        data = self.subject_data[subject_id][series_id]['data']
        events = self.subject_data[subject_id][series_id]['events']
        
        # Extract window
        end = min(window_start + window_length, len(data))
        window_data = data[window_start:end]
        window_events = events[window_start:end]
        
        # Visualize using the EEGActionVisualization class
        EEGActionVisualization.plot_eeg_with_actions(
            window_data, window_events, 
            title=f'Subject {subject_id}, Series {series_id}'
        )
        
        # Create action heatmap
        EEGActionVisualization.create_action_heatmap(
            window_events, 
            title=f'Action Heatmap - Subject {subject_id}, Series {series_id}'
        )
        
        # If classifier is available, make predictions and compare
        if self.classifier is not None:
            print("Making predictions with classifier...")
            window_preds = []
            
            # Process data in windows
            for start in range(0, len(window_data) - self.window_size, int(self.window_size * 0.5)):
                end = start + self.window_size
                if end > len(window_data):
                    break
                    
                segment = window_data[start:end]
                
                # Make prediction
                with torch.no_grad():
                    segment_tensor = torch.tensor(segment, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
                    segment_tensor = segment_tensor.to(self.device)
                    outputs = self.classifier(segment_tensor)
                    preds = outputs.cpu().numpy().squeeze()
                
                window_preds.append(preds)
            
            if window_preds:
                # Interpolate predictions to match original window length
                preds_array = np.array(window_preds)
                x_pred = np.linspace(0, 1, len(preds_array))
                x_full = np.linspace(0, 1, end - window_start)
                
                interpolated_preds = np.zeros((end - window_start, len(self.action_names)))
                for i in range(len(self.action_names)):
                    # Interpolate each action separately
                    interpolated_preds[:, i] = np.interp(x_full, x_pred, preds_array[:, i])
                
                # Plot comparison
                plt.figure(figsize=(15, 12))
                for i, action_name in enumerate(self.action_names):
                    plt.subplot(len(self.action_names), 1, i+1)
                    plt.plot(window_events[:, i], 'b-', label='True')
                    plt.plot(interpolated_preds[:, i], 'r-', label='Predicted')
                    plt.title(action_name)
                    plt.ylim([-0.1, 1.1])
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig('action_prediction_comparison.png')
                plt.close()
                
                print("Prediction comparison plot saved.")
        
        return window_data, window_events


def main():
    # Example usage
    data_dir = os.path.join('data', 'train')
    classifier_path = 'best_eeg_action_classifier.pth'
    
    # Create analyzer
    analyzer = ActionAnalyzer(
        data_dir=data_dir,
        classifier_path=classifier_path
    )
    
    # Load data (use first 3 subjects for demonstration)
    subject_ids = [1, 2, 3]
    analyzer.load_data(subject_ids=subject_ids)
    
    # Analyze action patterns
    results = analyzer.analyze_action_patterns()
    analyzer.plot_action_analysis(results)
    
    # Identify common action sequences
    top_sequences, all_sequences = analyzer.identify_action_sequences(min_sequence_length=5)
    analyzer.plot_common_sequences(top_sequences)
    
    # Evaluate classifier
    eval_results = analyzer.evaluate_classifier(test_subject_ids=[3])  # Use subject 3 as test
    
    # Visualize actions for a specific subject
    analyzer.visualize_subject_actions(subject_id='subj1', series_id='series1', window_start=0, window_length=1000)
    
    print("Action analysis complete!")


if __name__ == "__main__":
    main() 