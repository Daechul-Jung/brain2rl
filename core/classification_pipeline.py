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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import classification components
from models.classification.time_series_dataset import TimeSeriesDataset
from models.classification.action_classifier import ActionClassifier
from models.classification.data_utilities import (
    load_sensor_data, preprocess_multilabel, 
    create_train_val_loaders,create_test_loader, save_preprocessing_info
)
from models.classification.segment_utils import *


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
        
        self.logger.info(f"Classification Pipeline initialized on device: {self.device}")
    
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
        task = self.config.get('task', 'both')

        x_batch, y_batch, _ = next(iter(train_loader))  ### (Batch size, Channel(number of sensors used), window_size)
        n_channels = x_batch.shape[1]
        n_times = x_batch.shape[2]

        self.model = ActionClassifier(n_channels=n_channels, n_times=n_times,
                                      n_behavior_classes=num_behave, n_gesture_classes=num_gesture, task=task, device=self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['classifier_lr'])

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val = 0

        def acc_from_logits(outputs, targets):
            if task == 'behavior':
                logits = outputs['behavior_logits']; y = targets

            elif task == 'gesture':
                logits = outputs['gesture_logits']; y = targets
            
            else :
                logits = outputs['behavior_logits']; y = targets[0]
            _, pred = logits.max(1)

            return (pred == y).float().mean().item() * 100
        
        for epoch in range(self.config['classifier_epochs']):
            self.model.train()
            train_loss, train_acc, n_batch = 0.0, 0.0, 0
            for batch in train_loader:
                if len(batch) == 3:
                    x_batch, y_batch, _ = batch
                else:
                    x_batch, y_batch = batch
                x_batch = x_batch.to(self.device)
                targets = self._extract_targets(y_batch, task)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self._compute_loss(outputs, targets=targets, task= task)
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
                    val_acc += acc_from_logits(outputs, targets, task)
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
        
    def evaluate_and_dump(self, test_loader: DataLoader, encoders, dump_dir='output/classifier'):
        os.makedirs(dump_dir, exist_ok=True)
        task = self.config.get('task', 'both')
        pool_k = self.config.get('pool_k', None)
        self.model.eval()

        all_pred_behavior, all_pred_gesture = [], []
        all_true_behavior, all_true_gesture = [], []
        all_tokens = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    x_batch, y_batch, _ = batch
                else:
                    x_batch, y_batch = batch
                x_batch = x_batch.to(self.device)

                outputs = self.model(x_batch, return_tokens=True, pool_tokens_to=pool_k)
                if 'tokens' in outputs:
                    all_tokens.append(outputs['tokens'].cpu().numpy())

                if task in ('behavior','both'):
                    pb = outputs['behavior_logits'].argmax(dim=1).cpu().numpy()
                    all_pred_behavior.append(pb)
                    tb = (y_batch[0] if task=='both' else y_batch).cpu().numpy()
                    all_true_behavior.append(tb)
                if task in ('gesture','both'):
                    pg = outputs['gesture_logits'].argmax(dim=1).cpu().numpy()
                    all_pred_gesture.append(pg)
                    tg = (y_batch[1] if task=='both' else y_batch).cpu().numpy()
                    all_true_gesture.append(tg)

        result = {}
        if all_tokens:
            result['tokens'] = np.concatenate(all_tokens, axis=0)  # (N, K, 128) when pool_k set
        if task in ('behavior','both'):
            result['pred_behavior'] = np.concatenate(all_pred_behavior, axis=0)
            result['true_behavior'] = np.concatenate(all_true_behavior, axis=0)
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
    @torch.no_grad()
    def export_segment_tokens(self, csv_path: str, save_dir: str,
                              pool_k: int =16, 
                              group_key: str = 'sequence_id',
                              segment_by: str = 'behavior'):
        """
        Create fixed k tokens per contiguous action segment and save as NPZ

        segment by behvaior
        """

        os.makedirs(save_dir, exist_ok=True)
        X_raw, y_str, groups, df = load_sensor_data(csv_path, group_key)
        X = self.scaler.transform(X_raw)

        y_enc = {
            'behavior': self.encoders['behavior'].transform(y_str['behavior']),
            'gesture' : self.encoders['gesture'].transform(y_str['gesture'])
        }
        group_vals = groups.astype(str)
        subjects = df['subject'].astype(str).values
        seq_ids  = df[group_key].astype(str).values

        # 2) Build segments
        use_gesture = (segment_by == 'behavior_gesture')
        segs = contiguous_segments(group_vals, y_enc['behavior'], y_enc['gesture'] if use_gesture else None)
        if not segs:
            raise ValueError("No segments found. Check labels/grouping.")

        # 3) Run CNN trunk -> K tokens per segment
        self.model.eval()
        all_tok = []
        beh_ids, ges_ids, seg_len, seg_subject, seg_seq = [], [], [], [], []

        C = X.shape[1]
        for (s, e, gk, b, ge) in segs:
            x_seg = torch.from_numpy(X[s:e].astype(np.float32).T).unsqueeze(0).to(self.device)  # (1,C,T)
            tok = self.model.tokens_from_raw(x_seg, pool_tokens_to=pool_k)  # (1,K,128)
            all_tok.append(tok.squeeze(0).cpu().numpy())  # (K,128)
            beh_ids.append(b)
            ges_ids.append(ge)
            seg_len.append(e - s)
            seg_subject.append(subjects[s])
            seg_seq.append(seq_ids[s])

        tokens = np.stack(all_tok, axis=0)              # (N_segments, K, 128)
        beh_ids = np.array(beh_ids, dtype=np.int64)
        ges_ids = np.array(ges_ids, dtype=np.int64)
        seg_len = np.array(seg_len, dtype=np.int32)

        # 4) Prototypes per behavior (mean over time then mean over segments)
        seg_emb = tokens.mean(axis=1)                   # (N_segments, 128)
        nB = len(self.encoders['behavior'].classes_)
        beh_proto = np.zeros((nB, 128), dtype=np.float32)
        for b in range(nB):
            m = seg_emb[beh_ids == b]
            if len(m) > 0:
                beh_proto[b] = m.mean(axis=0)

        out = {
            'tokens': tokens,                           # (N_seg, K, 128)
            'behavior_id': beh_ids,                     # (N_seg,)
            'behavior_str': self.encoders['behavior'].inverse_transform(beh_ids),
            'gesture_id': ges_ids,                      # (N_seg,)
            'gesture_str': np.array(
                [self.encoders['gesture'].inverse_transform([g])[0]
                if g >=0 else '' for g in ges_ids], dtype=object),
            'segment_len': seg_len,                     # rows in original CSV
            'subject': np.array(seg_subject, dtype=object),
            group_key: np.array(seg_seq, dtype=object),
            'behavior_prototypes': beh_proto,           # (n_behavior,128)
            'pool_k': np.int32(pool_k)
        }

        save_path = os.path.join(save_dir, f'segment_tokens_K{pool_k}.npz')
        np.savez(save_path, **out)
        self.logger.info(f"Saved segment tokens to {save_path} "
                        f"(segments={tokens.shape[0]}, K={pool_k})")
        return save_path
    
    def run(self, train_csv, test_csv, output_dir='output'):
        # 1) load & preprocess
        X_train_raw, y_train_str, group_train, df_tr = load_sensor_data(train_csv, group_by='sequence_id')
        X_train, y_train, self.scaler, self.encoders = preprocess_multilabel(X_train_raw, y_train_str)

        # ---- choose training mode ----
        train_mode = self.config.get('train_mode', 'segments')  # 'segments' or 'windows'
        task = self.config.get('task', 'both')

        if train_mode == 'windows':
            # original windowed loaders (but group-aware inside your util)
            train_loader, val_loader = create_train_val_loaders(
                X_train=X_train, y_train_enc=y_train, groups_train=group_train,
                window_size=self.config['window_size'],
                batch_size=self.config['batch_size'],
                overlap=0.5, task=task
            )
        else:
            # --- SEGMENT MODE: build contiguous segments and split by groups ---
            use_gesture_for_seg = (self.config.get('segment_by','behavior') == 'behavior_gesture')
            segs_all = contiguous_segments(group_train.astype(str),
                                        y_train['behavior'],
                                        y_train['gesture'] if use_gesture_for_seg else None)

            # split by unique groups (e.g., sequence_id) to avoid leakage
            uniq_groups = np.array(sorted(set(group_train.astype(str))))
            gss = GroupShuffleSplit(test_size=0.2, random_state=42)
            g_tr_idx, g_va_idx = next(gss.split(uniq_groups, groups=uniq_groups))
            train_groups = set(uniq_groups[g_tr_idx]); val_groups = set(uniq_groups[g_va_idx])

            segs_tr = [s for s in segs_all if s[2] in train_groups]
            segs_va = [s for s in segs_all if s[2] in val_groups]

            ds_tr = SegmentDataset(X_train, y_train, group_train, segs_tr, task=task)
            ds_va = SegmentDataset(X_train, y_train, group_train, segs_va, task=task)

            train_loader = DataLoader(ds_tr, batch_size=self.config['batch_size'],
                                    shuffle=True, collate_fn=lambda b: pad_collate(b, task=task))
            val_loader   = DataLoader(ds_va, batch_size=self.config['batch_size'],
                                    shuffle=False, collate_fn=lambda b: pad_collate(b, task=task))

        num_behave = len(self.encoders['behavior'].classes_)
        num_gesture = len(self.encoders['gesture'].classes_)
        self.logger.info(f'num behavior: {num_behave} and num gesture: {num_gesture}')
        history = self.train(train_loader, val_loader, num_behave, num_gesture)

        save_preprocessing_info(self.scaler, self.encoders,
                                os.path.join(output_dir, 'classifier', 'preproc.pkl'))

        # 2) test
        X_test_raw, y_test_str, group_test, _ = load_sensor_data(test_csv, group_by='sequence_id')
        X_test = self.scaler.transform(X_test_raw)
        y_test = {k: self.encoders[k].transform(y_test_str[k]) for k in ['behavior','gesture']}

        if train_mode == 'windows':
            test_loader = create_test_loader(
                X_test=X_test, y_test_enc=y_test,
                window_size=self.config['window_size'],
                batch_size=self.config['batch_size'],
                overlap=0.5, task=task
            )
        else:
            # segment test loader
            use_gesture_for_seg = (self.config.get('segment_by','behavior') == 'behavior_gesture')
            segs_te = contiguous_segments(group_test.astype(str),
                                        y_test['behavior'],
                                        y_test['gesture'] if use_gesture_for_seg else None)
            ds_te = SegmentDataset(X_test, y_test, group_test, segs_te, task=task)
            test_loader = DataLoader(ds_te, batch_size=self.config['batch_size'],
                                    shuffle=False, collate_fn=lambda b: pad_collate(b, task=task))

        preds = self.evaluate_and_dump(test_loader=test_loader, encoders=self.encoders,
                                    dump_dir=os.path.join(output_dir,'classifier'))
        return {'history': history, 'preds': preds}
    

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
        'window_size': 100,
        'batch_size': 128,
        'classifier_lr': 1e-3,
        'classifier_epochs': 500,
        'classifier_dropout': 0.1,
        'task': 'both', 
        'train_mode': 'segments',    # <-- default to SEGMENTS
        'pool_k': 16,                # <-- fixed number of tokens per segment for training & eval
        'segment_by': 'behavior',
    }
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--subject-print', help='Subject ID from test.csv to print per-window predictions')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--export-segment-tokens', action='store_true',
                        help='Export fixed-K tokens per action segment')
    parser.add_argument('--export-from', choices=['train','test'], default='test',
                        help='Which split CSV to export from')
    parser.add_argument('--segment-by', choices=['behavior','gesture'], default='behavior')
    parser.add_argument('--group-key', default='sequence_id',
                        help="Group boundary key for segments (e.g., 'sequence_id' or 'subject')")
    parser.add_argument('--train-mode', choices=['segments','windows'], default='segments',
                        help='Train on contiguous action segments (variable length) or fixed windows')
    parser.add_argument('--pool-k', type=int, default=16,
                        help='Fixed number of tokens per segment after pooling (segments mode)')
    args = parser.parse_args()

    # ...
    cfg = create_classification_config()
    cfg['train_mode'] = args.train_mode
    cfg['pool_k'] = args.pool_k
    cfg['segment_by'] = args.segment_by
    # args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = create_classification_config()
    pipe = ClassificationOnlyPipeline(cfg)
    res = pipe.run(args.train_csv, args.test_csv, output_dir=args.output_dir)


    if args.export_segment_tokens:
        csv_src = args.train_csv if args.export_from == 'train' else args.test_csv
        pipe.export_segment_tokens(
            csv_path=csv_src,
            save_dir=os.path.join(args.output_dir, 'classifier'),
            pool_k=args.pool_k,
            group_key=args.group_key,
            segment_by=args.segment_by
        )
    if args.subject_print:
        pipe.print_subject_predictions(args.test_csv, args.subject_print)

    print("\n Done. Files:")
    print(" - Best classifier: output/classifier/best_classifier.pth")
    print(" - Test preds & tokens: output/classifier/test_predictions_and_tokens.npz")

if __name__ == '__main__':
    main()