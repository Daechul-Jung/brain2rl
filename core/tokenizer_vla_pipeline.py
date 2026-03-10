"""
VLA Tokenizer Pipeline
======================

Trains a CNN-based EEG tokenizer whose outputs can be fed as soft prompt tokens
into open-source language models (LLaMA, etc.) for action-language prediction.

Pipeline
--------
EEG Segment (C, T)
  → CNN Trunk (shared with ActionClassifier)     → (B, 128, T')
  → Adaptive Pool to K fixed tokens              → (B, 128, K)
  → Linear Projection to LLM embedding dim       → (B, K, llm_dim)
  → (Optional) Mean-pool + contrastive / classif.→  training signal

Two training modes:
  - 'classify'    : CrossEntropy on action-label prediction (fast, no LLM)
  - 'contrastive' : CLIP-style alignment to frozen text embeddings (richer)

Inference with LLM
------------------
After training, EEG tokens (B, K, llm_dim) can be inserted as prefix soft-prompts
before the instruction text embedding of any causal LLM that accepts embedding inputs.
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classification.action_classifier import ActionClassifier


# ---------------------------------------------------------------------------
# CNN EEG Encoder (reuses ActionClassifier trunk)
# ---------------------------------------------------------------------------

class EEGVLATokenizer(nn.Module):
    """
    Maps a variable-length EEG segment to K fixed soft-prompt tokens.

    Args:
        n_channels  : number of EEG channels (C)
        n_times     : representative sequence length (used to init trunk only)
        pool_k      : number of output tokens per segment
        token_dim   : dimension of each output token (default 128 – CNN trunk dim)
        llm_dim     : LLM embedding dimension for projection (e.g. 4096 for LLaMA-7B)
    """

    def __init__(self, n_channels: int, n_times: int,
                 pool_k: int = 16, token_dim: int = 128, llm_dim: int = 4096):
        super().__init__()
        self.pool_k = pool_k
        self.token_dim = token_dim
        self.llm_dim = llm_dim

        # Borrow only the CNN trunk from ActionClassifier
        _dummy = ActionClassifier(
            n_channels=n_channels, n_times=n_times,
            n_behavior_classes=2, n_gesture_classes=2,
            task='behavior'
        )
        self.trunk = _dummy.trunk   # Conv1D backbone  (B, 128, T')
        self.proj_token = _dummy.proj  # Linear(128, 128)

        # Project 128-dim CNN tokens → LLM embedding space
        self.to_llm = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, llm_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        Returns: cnn_tokens (B, pool_k, 128)
        """
        h = self.trunk(x)                                           # (B, 128, T')
        h = F.adaptive_avg_pool1d(h, self.pool_k)                  # (B, 128, K)
        tokens = self.proj_token(h.transpose(1, 2))                 # (B, K, 128)
        return tokens

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns both CNN-space tokens and LLM-projected tokens.
        x : (B, C, T)
        """
        cnn_tokens = self.encode(x)                                 # (B, K, 128)
        llm_tokens = self.to_llm(cnn_tokens)                       # (B, K, llm_dim)
        return {'cnn_tokens': cnn_tokens, 'llm_tokens': llm_tokens}


# ---------------------------------------------------------------------------
# Classification head used during training
# ---------------------------------------------------------------------------

class EEGClassifyHead(nn.Module):
    """Global mean-pool over K tokens → linear softmax for action labels."""

    def __init__(self, token_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(token_dim, n_classes)

    def forward(self, cnn_tokens: torch.Tensor) -> torch.Tensor:
        """cnn_tokens: (B, K, 128) → logits (B, n_classes)"""
        return self.fc(cnn_tokens.mean(dim=1))


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class VLATokenizerPipeline:
    """
    Trains EEGVLATokenizer so that its output tokens carry action-label semantics
    compatible with an LLM embedding space.

    Usage
    -----
    pipe = VLATokenizerPipeline(cfg)
    pipe.train(X_segments, y_labels)          # X: (N, C, T_seg)  y: (N,) int
    llm_tokens = pipe.get_llm_tokens(X_new)   # (N, K, llm_dim)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._make_logger()

        self.tokenizer: Optional[EEGVLATokenizer] = None
        self.cls_head: Optional[EEGClassifyHead] = None
        self.optimizer = None
        self.history: Dict[str, List] = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []
        }

    # ------------------------------------------------------------------
    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger('VLATokenizer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(h)
        return logger

    # ------------------------------------------------------------------
    def _build(self, n_channels: int, n_times: int, n_classes: int):
        pool_k   = self.cfg.get('pool_k', 16)
        llm_dim  = self.cfg.get('llm_dim', 4096)
        token_dim = 128

        self.tokenizer = EEGVLATokenizer(
            n_channels=n_channels, n_times=n_times,
            pool_k=pool_k, token_dim=token_dim, llm_dim=llm_dim
        ).to(self.device)

        self.cls_head = EEGClassifyHead(token_dim=token_dim, n_classes=n_classes).to(self.device)

        params = list(self.tokenizer.parameters()) + list(self.cls_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.get('lr', 1e-3),
                                           weight_decay=1e-4)

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List]:
        """
        Train the VLA tokenizer with a classification objective.

        Args
        ----
        X : EEG segments  (N, C, T)
        y : integer action labels  (N,)
        """
        n_classes = int(y.max()) + 1
        N, C, T = X.shape
        self._build(n_channels=C, n_times=T, n_classes=n_classes)

        # Split train/val
        idx = np.random.permutation(N)
        cut = int(N * 0.8)
        tr_idx, va_idx = idx[:cut], idx[cut:]

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))

        dl_tr = DataLoader(TensorDataset(X_t[tr_idx], y_t[tr_idx]),
                           batch_size=self.cfg.get('batch_size', 32), shuffle=True)
        dl_va = DataLoader(TensorDataset(X_t[va_idx], y_t[va_idx]),
                           batch_size=self.cfg.get('batch_size', 32), shuffle=False)

        criterion = nn.CrossEntropyLoss()
        epochs = self.cfg.get('epochs', 50)

        for ep in range(epochs):
            # --- train ---
            self.tokenizer.train(); self.cls_head.train()
            tr_loss = tr_acc = n_b = 0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.tokenizer(xb)
                logits = self.cls_head(out['cnn_tokens'])
                loss = criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.tokenizer.parameters()) + list(self.cls_head.parameters()), 1.0)
                self.optimizer.step()

                tr_loss += loss.item()
                tr_acc  += (logits.argmax(1) == yb).float().mean().item() * 100
                n_b += 1
            tr_loss /= max(1, n_b); tr_acc /= max(1, n_b)

            # --- val ---
            self.tokenizer.eval(); self.cls_head.eval()
            va_loss = va_acc = n_b = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.tokenizer(xb)
                    logits = self.cls_head(out['cnn_tokens'])
                    loss = criterion(logits, yb)
                    va_loss += loss.item()
                    va_acc  += (logits.argmax(1) == yb).float().mean().item() * 100
                    n_b += 1
            va_loss /= max(1, n_b); va_acc /= max(1, n_b)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(va_acc)
            self.logger.info(
                f"Epoch {ep+1}/{epochs}  "
                f"Train {tr_loss:.4f}/{tr_acc:.1f}%  Val {va_loss:.4f}/{va_acc:.1f}%"
            )

        return self.history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_llm_tokens(self, X: np.ndarray) -> np.ndarray:
        """
        Export LLM-aligned soft-prompt tokens.

        Args
        ----
        X : (N, C, T)

        Returns
        -------
        llm_tokens : (N, K, llm_dim)  – ready to prepend to LLM input embeddings
        """
        assert self.tokenizer is not None, "Call train() first."
        self.tokenizer.eval()
        all_tok = []
        bs = self.cfg.get('batch_size', 32)
        X_t = torch.from_numpy(X.astype(np.float32))
        for i in range(0, len(X_t), bs):
            xb = X_t[i:i+bs].to(self.device)
            all_tok.append(self.tokenizer(xb)['llm_tokens'].cpu().numpy())
        return np.concatenate(all_tok, axis=0)

    @torch.no_grad()
    def get_cnn_tokens(self, X: np.ndarray) -> np.ndarray:
        """
        Export 128-dim CNN tokens (before LLM projection).

        Returns
        -------
        cnn_tokens : (N, K, 128)
        """
        assert self.tokenizer is not None, "Call train() first."
        self.tokenizer.eval()
        all_tok = []
        bs = self.cfg.get('batch_size', 32)
        X_t = torch.from_numpy(X.astype(np.float32))
        for i in range(0, len(X_t), bs):
            xb = X_t[i:i+bs].to(self.device)
            all_tok.append(self.tokenizer(xb)['cnn_tokens'].cpu().numpy())
        return np.concatenate(all_tok, axis=0)

    def predict_action(self, X: np.ndarray, encoder=None) -> np.ndarray:
        """Run classification head to get action label predictions."""
        assert self.tokenizer is not None and self.cls_head is not None
        self.tokenizer.eval(); self.cls_head.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.tokenizer(X_t)
            preds = self.cls_head(out['cnn_tokens']).argmax(dim=1).cpu().numpy()
        if encoder is not None:
            return encoder.inverse_transform(preds)
        return preds

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        ckpt = {
            'cfg': self.cfg,
            'tokenizer': self.tokenizer.state_dict(),
            'cls_head': self.cls_head.state_dict(),
            'history': self.history,
        }
        torch.save(ckpt, path)
        self.logger.info(f"Saved VLA tokenizer to {path}")

    def load(self, path: str, n_channels: int, n_times: int, n_classes: int):
        ckpt = torch.load(path, map_location=self.device)
        self.cfg = ckpt['cfg']
        self._build(n_channels, n_times, n_classes)
        self.tokenizer.load_state_dict(ckpt['tokenizer'])
        self.cls_head.load_state_dict(ckpt['cls_head'])
        self.history = ckpt.get('history', self.history)
        self.logger.info(f"Loaded VLA tokenizer from {path}")

    # ------------------------------------------------------------------
    def inference_with_llm(self, X: np.ndarray, llm_model, tokenizer_hf,
                           prompt_template: str = "The robot should perform: ") -> List[str]:
        """
        Perform inference by prepending EEG soft-prompt tokens to a causal LLM.

        Requires a HuggingFace model that accepts `inputs_embeds`.
        Works with LLaMA, Mistral, and similar causal LMs.

        Args
        ----
        X : EEG segments (N, C, T)
        llm_model : HuggingFace causal LM (AutoModelForCausalLM)
        tokenizer_hf : HuggingFace tokenizer
        prompt_template : Text prompt appended after EEG soft tokens

        Returns
        -------
        predictions : list of generated strings
        """
        llm_tokens = self.get_llm_tokens(X)   # (N, K, llm_dim)
        llm_model.eval()
        predictions = []

        prompt_ids = tokenizer_hf(prompt_template, return_tensors='pt').input_ids.to(self.device)
        embed_layer = llm_model.get_input_embeddings()
        prompt_emb  = embed_layer(prompt_ids)  # (1, L_prompt, llm_dim)

        for i in range(len(X)):
            eeg_emb = torch.from_numpy(llm_tokens[i:i+1]).to(self.device)   # (1, K, llm_dim)
            combined = torch.cat([eeg_emb, prompt_emb], dim=1)              # (1, K+L, llm_dim)

            with torch.no_grad():
                out_ids = llm_model.generate(
                    inputs_embeds=combined,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9
                )
            text = tokenizer_hf.decode(out_ids[0], skip_special_tokens=True)
            predictions.append(text)

        return predictions


def _make_cfg(args) -> Dict[str, Any]:
    return {
        'pool_k':      args.pool_k,
        'llm_dim':     args.llm_dim,
        'lr':          args.lr,
        'batch_size':  args.batch_size,
        'epochs':      args.epochs,
    }


def main():
    import argparse
    from models.classification.eeg_raw_loader import load_eeg_dataset

    parser = argparse.ArgumentParser(description='Train VLA EEG Tokenizer')
    # Accept raw EEG directory (primary) or pre-processed CSV (legacy)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--eeg-dir',  help='Raw EEG .txt directory (recommended)')
    grp.add_argument('--train-csv', help='Pre-processed CSV (legacy)')
    parser.add_argument('--output-dir', default='output/tokenizer_vla')
    parser.add_argument('--pool-k',     type=int,   default=16)
    parser.add_argument('--llm-dim',    type=int,   default=4096,
                        help='LLM embedding dim (4096=LLaMA-7B, 2048=smaller)')
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--window-sec', type=float, default=1.0)
    parser.add_argument('--overlap',    type=float, default=0.5)
    parser.add_argument('--labeling',   choices=['block', 'excel'], default='block')
    parser.add_argument('--excel-path', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.eeg_dir:
        X, y_dict, groups, meta = load_eeg_dataset(
            data_dir=args.eeg_dir,
            window_sec=args.window_sec,
            overlap=args.overlap,
            labeling=args.labeling,
            excel_path=args.excel_path,
        )
        y_behavior = y_dict['behavior']
    else:
        from models.classification.data_utilities import (
            load_sensor_data, preprocess_multilabel)
        X_raw, y_str, groups, df = load_sensor_data(args.train_csv,
                                                     group_by='sequence_id')
        X, y_enc, scaler, encoders = preprocess_multilabel(X_raw, y_str)
        y_behavior = y_enc['behavior']
        if X.ndim == 2:
            C = X_raw.shape[1];  T = X.shape[1] // C
            X = X.reshape(-1, C, T)

    cfg = _make_cfg(args)
    pipe = VLATokenizerPipeline(cfg)
    model_path = args.model_path or os.path.join(args.output_dir, 'vla_tokenizer.pth')

    history = pipe.train(X, y_behavior)
    pipe.save(model_path)

    tokens = pipe.get_cnn_tokens(X)
    np.save(os.path.join(args.output_dir, 'cnn_tokens.npy'), tokens)
    print(f"CNN tokens shape: {tokens.shape}")
    print(f"Best val acc: {max(history['val_acc']):.1f}%")
    print(f"Model saved to: {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_TOKEN = os.getenv("HF_TOKEN")
    hf_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token="hf_xxx")
    llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                                token=HF_TOKEN, device_map="auto")

    preds = pipe.inference_with_llm(X[:5], llm, hf_tok,
                                    prompt_template="The robot should perform: ")
    for p in preds:
        print(p)


if __name__ == '__main__':
    main()
