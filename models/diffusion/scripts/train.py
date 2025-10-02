import argparse
from pathlib import Path
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.models.unet import UNetDenoiser
from models.diffusion.diffusion import Diffusion
from models.diffusion.trainer import Trainer
from models.diffusion.data.cifar10 import make_cifar10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs", type=str, default="runs")
    args = p.parse_args()


    device = torch.device(args.device)
    loader = make_cifar10(args.data, args.bs)
    model = UNetDenoiser(in_channels=3, base=128, time_dim=256, cond_dim=0, attn_res=[16])
    diff = Diffusion(T=args.T, device=device.type)
    tr = Trainer(model, diff, lr=args.lr, ema_decay=args.ema, device=device)
    tr.train(loader, args.epochs, Path(args.runs))
    print("Training finished. Checkpoint saved at runs/last.pt")


if __name__ == "__main__":
    main()

