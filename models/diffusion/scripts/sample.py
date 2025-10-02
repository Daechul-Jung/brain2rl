
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.models.unet import UNetDenoiser
from models.diffusion.diffusion import Diffusion
from models.diffusion.schdules import build_constants


def save_samples(x: torch.Tensor, out: Path, nrow: int = 8):
    x = (x.clamp(-1, 1) + 1) / 2.0
    save_image(x, str(out), nrow=nrow)




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/last.pt")
    p.add_argument("--num", type=int, default=64)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="samples.png")
    args = p.parse_args()


    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNetDenoiser(in_channels=3, base=128, time_dim=256, cond_dim=0, attn_res=[16]).to(device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:
        model.load_state_dict({**model.state_dict(), **ckpt["ema"]})
    model.eval()


    diff = Diffusion(T=ckpt["diff"]["T"], device=device.type)
    diff.betas = ckpt["diff"]["betas"].to(device)
    diff.alphas, diff.alpha_bar, diff.sqrt_ab, diff.sqrt_1mab = build_constants(diff.betas)


    shape = (args.num, 3, 32, 32)
    with torch.no_grad():
        x = diff.sample_ddim(model, shape, cond=None, steps=args.ddim_steps, eta=args.eta, cfg_scale=args.cfg_scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_samples(x, out)
    print(f"Saved {args.num} samples to {out}")


if __name__ == "__main__":
    main()