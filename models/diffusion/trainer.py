import os, sys
from pathlib import Path
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.utils.ema import *

class Trainer:
    def __init__(self, model: nn.Module, diffusion, lr = 2e-4, ema_decay : float = 0.0, device = 'cuda'):
        self.device = device
        self.model = model.to(self.device)  ## Unet denoiser
        self.diff = diffusion
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas = (0.9, 0.99), weight_decay=1e-4)
        self.ema = EMA(self.model, ema_decay) if ema_decay > 0 else None

    def loss_step(self, x0: torch.Tensor, cond_vec: torch.Tensor | None = None, cfg_drop_p: float = 0.1):
        b = x0.size(0)
        t = torch.randint(0, self.diff.T, (b, ), device=self.device)
        noise = torch.randn_like(x0) 
        x_t = self.diff.q_sample(x0, t, noise)  ### Same size with x0
        # classifier-free guidance training: randomly drop cond
        if cond_vec is not None and torch.rand(1, device=self.device) < cfg_drop_p:
          cond_vec = None
        eps_pred = self.model(x_t, t, cond_vec)  ## cond = none
        return F.mse_loss(eps_pred, noise)


    def train(self, loader: DataLoader, epochs: int, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        step = 0
        
        for ep in range(1, epochs + 1):
            pbar = tqdm(loader, total=len(loader), desc=f"epoch {ep}/{epochs}", leave=True)
            for batch in pbar:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x0, cond_vec = batch
                    cond_vec = cond_vec.to(self.device)
                else:
                    x0, cond_vec = batch[0], None
                    
                self.model.train()
                x = x.to(self.device)
                loss = self.loss_step(x0=x, cond=None)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.ema:
                    self.ema.update(self.model)
                pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)
                step += 1
            # save checkpoint each epoch
            ckpt = {
            "model": self.model.state_dict(),
            "ema": (self.ema.shadow if self.ema else None),
            "diff": {"T": self.diff.T, "betas": self.diff.betas},
            }
            torch.save(ckpt, outdir / "last.pt")
