import argparse
from pathlib import Path
import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.models.unet import UNetDenoiser
from models.diffusion.diffusion import Diffusion
from models.diffusion.schdules import build_constants  

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs_action/last.pt")
    p.add_argument("--num", type=int, default=64)
    p.add_argument("--H", type=int, default=1)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--ddim_steps", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="samples_actions.pt")
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNetDenoiser(
        action_dim=args.action_dim, base=128, time_dim=256, cond_dim=args.cond_dim, attn_lens=[8,16,32]
    ).to(device)

    model.load_state_dict(ckpt["model"], strict = False)
    ema_shadow = ckpt.get('ema')

    if ema_shadow is not None:
        model.load_state_dict({**model.state_dict(), **ema_shadow}, strict = False)

    model.eval()

    diffusion = Diffusion(T= ckpt["diff"]["T"], device=device.type)
    diffusion.betas = ckpt["diff"]["betas"].to(device)
    diffusion.alphas, diffusion.alpha_bar, diffusion.sqrt_alpha_bar, diffusion.sqrt_1mab = build_constants(diffusion.betas)

    cond_vec = torch.zeros(args.num, args.cond_dim, device=device)
    shape = (args.num, args.H, args.action_dim)
    with torch.no_grad():
        # Non-differentiable sampler for *offline* generation / inspection
        x = diffusion.sample_ddim(model, shape, cond=cond_vec, steps=args.ddim_steps, eta=args.eta, cfg_scale=args.cfg_scale)

    # Save tensor; for H==1 this is delta-action per sample
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.cpu(), out_path)
    # also save as numpy if you want
    np.save(str(out_path.with_suffix(".npy")), x.cpu().numpy())

    print(f"Saved {args.num} action samples to {out_path} and {out_path.with_suffix('.npy')}")

if __name__ == "__main__":
    main()