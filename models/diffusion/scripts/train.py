
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
import os, sys
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.models.unet import UNetDenoiser
from models.diffusion.diffusion import Diffusion
from models.diffusion.schdules import build_constants
from models.diffusion.trainer import Trainer

def save_samples(x: torch.Tensor, out: Path, nrow: int = 8):
    x = (x.clamp(-1, 1) + 1) / 2.0
    save_image(x, str(out), nrow=nrow)

class DummyActionCondDataset(Dataset):
    """
    Needs to be replaced with real dataset
    """
    def __init__(self, n=5000, H=1, action_dim=7, cond_dim=256, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x0 = torch.randn(n, H, action_dim, generator=g) * 0.5
        self.cond = torch.randn(n, cond_dim, generator=g)

    def __len__(self): return self.x0.shape[0]
    def __getitem__(self, i):
        return self.x0[i], self.cond[i]

def main():
    p = argparse.ArgumentParser()
    # — core dimensions —
    p.add_argument("--action_dim", type=int, default=7)      # robot DoF or action size
    p.add_argument("--H", type=int, default=1)               # horizon; 1 == delta-action
    p.add_argument("--cond_dim", type=int, default=256)      # from tokens/state encoder
    p.add_argument("--base", type=int, default=128)          # UNet base channels
    p.add_argument("--time_dim", type=int, default=256)      # time embedding dim
    p.add_argument("--attn_lens", type=str, default="8,16,32")  # attention at these L

    # — training —
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs", type=str, default="runs_action")
    p.add_argument("--num_train", type=int, default=50000)   # replace with your dataset size
    args = p.parse_args()

    device = torch.device(args.device)
    attn_lens = tuple(int(x) for x in args.attn_lens.split(",")) if args.attn_lens else ()
    ds = DummyActionCondDataset(n=args.num_train, H=args.H, action_dim=args.action_dim, cond_dim=args.cond_dim)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # === Model ===
    model = UNetDenoiser(
        action_dim=args.action_dim,
        base=args.base,
        time_dim=args.time_dim,
        cond_dim=args.cond_dim,
        attn_lens=list(attn_lens)
    ).to(device)

    # === Diffusion ===
    diff = Diffusion(T=args.T, device=device.type)

    # === Trainer ===
    tr = Trainer(model=model, diffusion=diff, lr=args.lr, ema_decay=args.ema, device=device)
    outdir = Path(args.runs)
    tr.train(loader, args.epochs, outdir)
    print(f"Training finished. Checkpoint saved at {outdir/'last.pt'}")

if __name__ == "__main__":
    main()
