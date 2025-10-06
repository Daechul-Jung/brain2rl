"""
Tokenization Pipeline
=====================

This module handles the tokenization of classified sensor data into tokens
with Query/Key/Value matrices for reinforcement learning trajectory control.

Author: Daechul Jung
Version: 1.0.0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.tokenization.brain_tokenizer_transformer import BrainTokenizer
from models.tokenization.transformer import *



class TokenizationPipeline:
    """
    Consume classifier tokens, contextualizes via transformer, emit Q/K/V + RL packs
    """

    def __init__(self, model_config: Dict[str, Any]):
        self.cfg = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('Brain2RL.Tokenization')
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

        self.tokenizer = None 
        self.attn = None 
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.is_trained = False
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def initialize_models(self, input_channels: int, input_length: int):
        self.tokenizer = BrainTokenizer(
            input_channels=input_channels,
            input_length=input_length,
            n_tokens=self.cfg['n_tokens'],
            embedding_dim=self.cfg['embedding_dim'],
            nhead=self.cfg['nhead'],
            num_encoder_layers=self.cfg['num_encoder_layers'],
            dropout=self.cfg['dropout'],
            use_conv_frontend=False
        ).to(self.device)

        self.attn = MultiHeadAttention(
            d_model=self.cfg['embedding_dim'],
            n_heads=self.cfg['nhead'],
            dropout = self.cfg['dropout']
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.tokenizer.parameters()) + list(self.attn.parameters()),
            lr = self.cfg.get('learning_rate', 1e-3)
        )

        self.logger.info(f"Initialized tokenizer (E={self.cfg['embedding_dim']}) with conv_frontend=False")
        

    def _spilt_train_val(self, dataset: Dataset, ratio:float = 0.8):
        n = len(dataset)
        n_train = int(n * ratio) 
        idx = np.arange(n); np.random.seed(42); np.random.shuffle(idx)

        train_idx, val_idx = idx[:n_train], idx[n_train:]

        tr = torch.utils.data.Subset(dataset, train_idx)
        va = torch.utils.data.Subset(dataset, val_idx)
        dl_tr = DataLoader(tr, batch_size=self.cfg.get('batch_size', 16), shuffle=True)
        dl_va = DataLoader(va, batch_size=self.cfg.get('batch_size', 16), shuffle=False)
        return dl_tr, dl_va   

    def _train_on_classifier_tokens(self, token_wins: np.ndarray, actions: np.ndarray, epochs:int = 500):
        """
        tokens_win: (N_windows, T', 128) from classifier stage
        actions:    (N_windows,) chosen labels 
        """


        dataset = TokensEpisodeDataset(
            tokens_win=token_wins, actions=actions,
            windows_per_episode=self.cfg.get('windows_per_episode', 8),
            stride_window=self.cfg.get('stride_windows', 8)
        )

        if len(dataset) == 0:
            raise ValueError("Too few windows to form episodes. Lower windows_per_episode or provide more data.")
        
        x_batch, y_batch = dataset[0]
        C, T = x_batch.shape
        self.initialize_models(input_channels=C, input_length=T)
        train_data, val_data = self._spilt_train_val(dataset, ratio=0.8)

        for ep in range(epochs):
            self.tokenizer.train()
            self.attn.train()
            train_loss = train_acc = num_batch = 0

            for x_batch, y_batch in tqdm(train_data, desc=f"Tok Epoch {ep+1}/{epochs}"):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                tok_logits = self.tokenizer(x_batch)  ## (Batch, S, n_tokens) teacher
                enc = self.tokenizer.encode(x_batch)  ## (Batch, sequence, E) embeddings

                attn_output, attn_info = self.attn(enc, enc, enc)
                print(f'shape of encoded tokens: {enc.shape} and shape of attention output: {attn_output.shape}')
                if not hasattr(self, 'prediction_head'):
                    self.prediction_head = nn.Linear(self.cfg['embedding_dim'], tok_logits.size(-1)).to(self.device)


                pred = self.prediction_head(attn_output[:, :-1, :]) ## (B, S-1, n_tokens)
                target = tok_logits [:, 1:, :].detach().argmax(dim = -1)  ## (B, S-1)
                loss_main = self.criterion(pred.reshape(-1, pred.size(-1)), target.reshape(-1))

                if not hasattr(self, 'action_head'):
                    self.action_head = nn.Linear(self.cfg['embedding_dim'], int(actions.max())+1).to(self.device)
                global_emb = enc.mean(dim=1)                     # (B, E)
                logits_action = self.action_head(global_emb)     # (B, A)
                loss_aux = self.criterion(logits_action, y_batch)

                loss = loss_main + self.cfg.get('aux_weight', 0.5) * loss_aux
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    acc = (logits_action.argmax(dim=1) == y_batch).float().mean().item() * 100.0
                train_loss += loss.item(); train_acc += acc; num_batch += 1

            train_loss /= max(1, n_b); train_acc /= max(1, n_b)

            # Val
            self.tokenizer.eval(); self.attn.eval()
            va_loss = va_acc = n_b = 0
            with torch.no_grad():
                for x_batch, y_batch in val_data:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    tok_logits = self.tokenizer(x_batch)
                    enc = self.tokenizer.encode(x_batch)
                    attn_output, attn_info = self.attn(enc, enc, enc)
                    if not hasattr(self, 'prediction_head'):
                        self.prediction_head = nn.Linear(self.cfg['embedding_dim'], tok_logits.size(-1)).to(self.device)
                    pred = self.prediction_head(attn_output[:, :-1, :])
                    tgt  = tok_logits[:, 1:, :].detach().argmax(dim=-1)
                    loss_main = self.criterion(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1))
                    loss = loss_main
                    if hasattr(self, 'action_head'):
                        global_emb = enc.mean(dim=1)
                        logits_action = self.action_head(global_emb)
                        loss_aux = self.criterion(logits_action, y_batch)
                        loss = loss_main + self.cfg.get('aux_weight', 0.5) * loss_aux
                        va_acc += (logits_action.argmax(dim=1) == y_batch).float().mean().item() * 100.0
                    
                    va_loss += loss.item(); n_b += 1
            va_loss /= max(1, n_b); va_acc /= max(1, n_b)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(va_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(va_acc)
            self.logger.info(f"[Tok] Epoch {ep+1}: Train {train_loss:.4f}/{train_acc:.2f}% | Val {va_loss:.4f}/{va_acc:.2f}%")

        self.is_trained = True
        return self.history

    @torch.no_grad()
    def build_rl_trajectories(self, tokens_win: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert windows -> episodes -> contextual embeddings + Q/K/V for RL.
        Returns a dict with padded tensors per batch (numpy arrays).
        """
        ds = TokensEpisodeDataset(
            tokens_win, actions,
            windows_per_episode=self.cfg.get('windows_per_episode', 8),
            stride_windows=self.cfg.get('stride_windows', 8)
        )
        if len(ds) == 0:
            raise ValueError("Too few windows to form episodes.")

        # collate all episodes (no batching needed for export)
        obs_embeddings = []  # (T, E)
        acts = []
        q_list, k_list, v_list, attn_w_list = [], [], [], []
        lens = []

        for i in tqdm(range(len(ds)), desc="Building RL Trajectories"):
            xb, yb = ds[i]   ### xb: (128, T)
            xb = xb.unsqueeze(0).to(self.device)   # (1, 128, T)

            enc = self.tokenizer.encode(xb)    ## (1, Tenc, E)
            attn_out, info = self.attn(enc, enc, enc)   # (1, Tenc, E), dict

            Tenc = enc.size(1)
            obs_embeddings.append(enc.squeeze(0).cpu().numpy())  # (Tenc, E)
            acts.append(int(yb.item()))
            q_list.append(info['query'].squeeze(0).cpu().numpy())         # (H, Tenc, D)
            k_list.append(info['key'].squeeze(0).cpu().numpy())
            v_list.append(info['value'].squeeze(0).cpu().numpy())
            attn_w_list.append(info['attention_weights'].squeeze(0).cpu().numpy())  # (H, Tenc, Tenc)
            lens.append(Tenc)

        # pad to the max length for easy batching
        max_len = max(lens)
        E = self.cfg['embedding_dim']; H = self.cfg['nhead']; D = E // H

        def pad_list(mats, pad_shape, axis=0, value=0.0):
            out = np.full((len(mats),) + pad_shape, value, dtype=np.float32)
            for i, m in enumerate(mats):
                sl = [slice(None)] * len(pad_shape)
                sl[0] = slice(0, m.shape[0])  # time
                out[(i, *sl)] = m
            return out

        # obs: (N_eps, max_len, E), mask: (N_eps, max_len)
        obs = np.full((len(obs_embeddings), max_len, E), 0.0, dtype=np.float32)
        mask = np.zeros((len(obs_embeddings), max_len), dtype=np.float32)
        for i, emb in enumerate(obs_embeddings):
            L = emb.shape[0]
            obs[i, :L] = emb
            mask[i, :L] = 1.0

        # Q/K/V: store as (N_eps, H, max_len, D) with padding
        def pad_qkv(lst, head_dims):
            out = np.zeros((len(lst),) + head_dims, dtype=np.float32)
            for i, m in enumerate(lst):
                H, L, D = m.shape
                out[i, :, :L, :] = m
            return out

        Q = pad_qkv(q_list, (H, max_len, D))
        K = pad_qkv(k_list, (H, max_len, D))
        V = pad_qkv(v_list, (H, max_len, D))

        # attention weights: (N_eps, H, max_len, max_len)
        Attn = np.zeros((len(attn_w_list), H, max_len, max_len), dtype=np.float32)
        for i, m in enumerate(attn_w_list):
            H_, L, _ = m.shape
            Attn[i, :, :L, :L] = m

        return {
            'obs_embeddings': obs,   # (N, T, E)
            'actions': np.array(acts, dtype=np.int64),   # (N,)
            'mask': mask,            # (N, T)
            'lengths': np.array(lens, dtype=np.int32),
            'Q': Q, 'K': K, 'V': V,  # (N, H, T, D)
            'attention': Attn        # (N, H, T, T)
        }

    def save_models(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'cfg': self.cfg,
            'tokenizer': self.tokenizer.state_dict(),
            'attn': self.attn.state_dict(),
            'history': self.history,
            'is_trained': self.is_trained
        }
        if hasattr(self, 'action_head'):
            data['action_head'] = self.action_head.state_dict()
        torch.save(data, path)
        self.logger.info(f"Saved tokenizer models to {path}")

    def load_models(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.cfg = data['cfg']
        # size models
        self.initialize_models(inp_channels=128, inp_length=self.cfg.get('max_sequence_length', 1000))
        self.tokenizer.load_state_dict(data['tokenizer'])
        self.attn.load_state_dict(data['attn'])
        if 'action_head' in data:
            self.action_head = nn.Linear(self.cfg['embedding_dim'], self.cfg.get('n_actions', 64)).to(self.device)
            self.action_head.load_state_dict(data['action_head'])
        self.history = data.get('history', self.history)
        self.is_trained = data.get('is_trained', False)
        self.logger.info(f"Loaded tokenizer models from {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tokenization from classifier tokens')
    parser.add_argument('--classifier-npz', required=True,
                        help='Path to output/classifier/test_predictions_and_tokens.npz')
    parser.add_argument('--mode', choices=['train', 'emit_rl'], default='train')
    parser.add_argument('--model-path', type=str, default='output/tokenizer/tokenizer.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--n-tokens', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--windows-per-episode', type=int, default=8)
    parser.add_argument('--stride-windows', type=int, default=8)
    parser.add_argument('--aux-weight', type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs('output/tokenizer', exist_ok=True)

    cfg = {
        'embedding_dim': args.embedding_dim,
        'n_tokens': args.n_tokens,
        'nhead': args.nhead,
        'num_encoder_layers': args.layers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'max_sequence_length': 4096,          # upper bound for adaptive pooling
        'windows_per_episode': args.windows_per_episode,
        'stride_windows': args.stride_windows,
        'aux_weight': args.aux_weight
    }

    data = np.load(args.classifier_npz)
    tokens_win = data['tokens'].astype(np.float32)  # (N_win, T', 128)
    # Use predicted behavior as the default action signal for RL conditioning
    actions = data['pred_behavior'] if 'pred_behavior' in data else np.zeros(len(tokens_win), dtype=np.int64)

    pipe = TokenizationPipeline(cfg)

    if args.mode == 'train':
        hist = pipe._train_on_classifier_tokens(tokens_win, actions, epochs=args.epochs)
        pipe.save_models(args.model_path)
        print(f"Training done. Last val acc: {hist['val_acc'][-1]:.2f}%")
    else:
        # Load and emit RL-ready pack
        pipe.load_models(args.model_path)
        rl = pipe.build_rl_trajectories(tokens_win, actions)
        np.savez('output/tokenizer/rl_trajectories.npz', **rl)
        print("Saved RL trajectories to output/tokenizer/rl_trajectories.npz")

if __name__ == '__main__':
    main()