#!/usr/bin/env python3
"""
Classification-Only Pipeline Runner

This script runs only the classification part of the Brain2RL pipeline,
skipping tokenization and RL state creation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import classification components
from models.classification.time_series_dataset import TimeSeriesDataset
from models.classification.action_classifier import ActionClassifier
from models.classification.data_utilities import (
    load_sensor_data, preprocess_multilabel, 
    create_train_val_loaders,create_test_loader, save_preprocessing_info
)


class ClassificationOnlyPipeline:    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logger()
        
        self.model = None
        self.scaler = None
        self.encoders = None 
        self.optimizer = None 

        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Classification-Only Pipeline initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('ClassificationOnlyPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _compute_loss(self, outputs, targets, task='both'):
        if task == 'behavior':
            return self.criterion(outputs['behavior_logits'], targets)
        
        elif task == 'gesture':
            return self.criterion(outputs['gesture_logits'], targets)
        
        y_behavior, y_gesture= targets

        loss_behavior = self.criterion(outputs['behavior_logits'], y_behavior)
        loss_gesture = self.criterion(outputs['gesture_logits'], y_gesture)

        return loss_behavior + loss_gesture
    
    def _extract_targets(self, batch_y, task = 'both'):
        if task == 'both':
            y_behavior, y_gesture = batch_y
            return (y_behavior.to(self.device), y_gesture.to(self.device))
        else:
            return batch_y.to(self.device)
    
    
    def train(self, train_loader, val_loader, num_behave, num_gesture):
        n_channels = train_loader.dataset.data.shape[0]
        n_times = self.config['window_size']
        task = self.config.get('task', 'both')

        self.model = ActionClassifier(n_channels=n_channels, n_times=n_times,
                                      n_behavior_classes=num_behave, n_gesture_classes=num_gesture, task=task)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['classifier_lr'])

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val = 0

        def acc_from_logits(outputs, targets):
            if task == 'behavior':
                logits = outputs['behavior_logits']; y = targets

            elif task == 'gesture':
                logits = outputs['gesture_logtis']; y = targets
            
            else :
                logits = outputs['behavior_logits']; y = targets[0]
            _, pred = logits.max(1)

            return (pred == y).float().mean().item() * 100
        
        for epoch in range(self.config['classifier_epoch']):
            self.model.train()
            train_loss, train_acc, n_batch = 0.0, 0.0, 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                targets = self._extract_targets(y_batch, task)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self._compute_loss(outputs, targets=targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_acc += acc_from_logits(outputs, targets)
                n_batch += 1

            train_loss /= max(1, n_batch)
            train_acc /= max(1, n_batch)

            self.model.eval()

            val_loss, val_acc, n_batch = 0.0, 0.0, 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    targets = self._extract_targets(y_batch, task)

                    outputs = self.model(x_batch)
                    loss = self._compute_loss(outputs, targets, task)

                    val_loss += loss.item()
                    val_acc += acc_from_logits(outputs, targets)
                    n_batch += 1
                val_loss /= max(1, n_batch)
                val_acc /= max(1, n_batch)

                history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)
                self.logger.info(f"Epoch {epoch+1}: Train {train_loss:.4f}/{train_acc:.2f}% | Val {val_loss:.4f}/{val_acc:.2f}%")

                if val_acc > best_val:
                    best_val = val_acc
                    os.makedirs('output/classifier', exist_ok=True)
                    torch.save(self.model.state_dict(), 'output/classifier/best_classifier.pth')
                    self.logger.info(f"Saved best classifier (val acc {best_val:.2f}%)")

        return history
        
    def evaluate_and_dump(self, test_loader: DataLoader, encoders, dump_dir = 'output/classifier'):
        os.makedirs(dump_dir, exist_ok= True)
        task = self.config.get('task', 'both')
        self.model.eval()
        all_pred_behavior, all_pred_gesture, all_true_behavior, all_true_gesture = [], [], [], []

        all_tokens = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch, return_tokens =True)
                token = outputs['tokens'].cpu().numpy()
                all_tokens.append(token)

                if task in ('behavior', 'both'):
                    predicted_behavior = outputs['behavior_logits'].argmax(dim = 1).cpu().numpy()
                    all_pred_behavior.append(predicted_behavior)
                    all_true_behavior.append((y_batch[0] if task == 'both' else y_batch).numpy())
                if task in ('gesture', 'both'):
                    predicted_gesture = outputs['gesture_logits'].argmax(dim = 1).cpu().numpy()
                    all_pred_gesture.append(predicted_gesture)
                    all_true_gesture.append((y_batch[1] if task == 'both'else y_batch).numpy())

        result = {'tokens': np.concatenate(all_tokens, axis=0)}
        if task in ('behavior','both'):
            result['pred_behavior'] = np.concatenate(all_pred_behavior, axis=0)
            result['true_behavior'] = np.concatenate(all_true_behavior, axis=0)
            # decode to strings for convenience
            result['pred_behavior_str'] = encoders['behavior'].inverse_transform(result['pred_behavior'])
            result['true_behavior_str'] = encoders['behavior'].inverse_transform(result['true_behavior'])
        if task in ('gesture','both'):
            result['pred_gesture'] = np.concatenate(all_pred_gesture, axis=0)
            result['true_gesture'] = np.concatenate(all_true_gesture, axis=0)
            result['pred_gesture_str'] = encoders['gesture'].inverse_transform(result['pred_gesture'])
            result['true_gesture_str'] = encoders['gesture'].inverse_transform(result['true_gesture'])

        np.savez(os.path.join(dump_dir, 'test_predictions_and_tokens.npz'), **result)
        self.logger.info(f"Saved predictions + tokens to {dump_dir}")

        return result
    def print_subject_predictions(self, test_csv: str, subject_id: str):
        """
        Load test.csv subset for a subject, run the classifier, and print per-window predicted actions.
        """
        
        X_raw, y_raw, groups, df = load_sensor_data(test_csv, subjects=[subject_id], group_by="sequence_id")
        # transform with train scaler/encoders
        X = self.scaler.transform(X_raw)
        y_enc = {k: self.encoders[k].transform(y_raw[k]) for k in ['behavior','gesture']}

        loader = create_test_loader(
            X, y_enc,
            window_size=self.config['window_size'],
            batch_size=self.config['batch_size'],
            overlap=0.5, task=self.config.get('task','both')
        )

        preds = self.evaluate_and_dump(loader, self.encoders)  # reuse forward pass

        # Print nicely
        beh = preds.get('pred_behavior_str')
        ges = preds.get('pred_gesture_str')
        print(f"\n=== Predictions for subject {subject_id} (windowed) ===")
        for i in range(len(beh) if beh is not None else len(ges)):
            btxt = beh[i] if beh is not None else "-"
            gtxt = ges[i] if ges is not None else "-"
            print(f"[win {i:04d}] behavior={btxt}  gesture={gtxt}")

    

    
    def run(self, train_csv, test_csv, output_dir='output'):
        X_train_raw, y_train_str, group_train, df = load_sensor_data(train_csv, group_by='sequence_id')
        X_train, y_train, self.scaler, self.encoders = preprocess_multilabel(X_train_raw, y_train_str)

        train_loader, val_loader = create_train_val_loaders(
            X_train=X_train, y_train_enc=y_train, groups_train=group_train, 
            window_size=self.config['window_size'],
            batch_size=self.config['batch_size'],
            overlap=0.5, task= self.config.get('task', 'both')
            )
        num_behave = len(self.encoders['behavior'].classes_)
        num_gesture = len(self.encoders['gesture'].classes_)
        print(f'num behavior: {num_behave} and num gesture: {num_gesture}')
        history = self.train(train_loader, val_loader, num_behave, num_gesture)

    def plot_training_history(self, history: Dict[str, List[float]], save_path: str):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Training history plot saved to {save_path}")


def create_classification_config() -> Dict[str, Any]:
    return {
        'window_size': 30,
        'batch_size': 32,
        'classifier_lr': 1e-3,
        'classifier_epochs': 50,
        'classifier_dropout': 0.3,
        'task': 'both',  # 'behavior' | 'gesture' | 'both'
    }
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--subject-print', help='Subject ID from test.csv to print per-window predictions')
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = create_classification_config()
    pipe = ClassificationOnlyPipeline(cfg)
    res = pipe.run(args.train_csv, args.test_csv, output_dir=args.output_dir)

    if args.subject_print:
        pipe.print_subject_predictions(args.test_csv, args.subject_print)

    print("\n Done. Files:")
    print(" - Best classifier: output/classifier/best_classifier.pth")
    print(" - Test preds & tokens: output/classifier/test_predictions_and_tokens.npz")

if __name__ == '__main__':
    main()