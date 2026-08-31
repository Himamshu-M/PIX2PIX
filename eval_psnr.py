import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from your training script
from pix2pix_tir2rgb import TIR2RGBDataset, GeneratorUNet,psnr


def psnr_full(pred, target):
    """Mean PSNR in dB for tensors in [-1, 1] (peak-to-peak range is 2)."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    # Clamp to avoid division by zero if perfect match
    mse = torch.clamp(mse, min=1e-8)
    return (10 * torch.log10(4.0 / mse)).mean().item()


def evaluate_psnr(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load generator
    G = GeneratorUNet(in_channels=1, out_channels=3).to(device)
    G.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    G.eval()

    # Dataset: either a dedicated test folder or the val split
    if args.test_root:
        ds = TIR2RGBDataset(
            args.test_root,
            args.rgb_dir,
            args.tir_dir,
            split="all",
            img_size=args.img_size
        )
    else:
        ds = TIR2RGBDataset(
            args.data_root,
            args.rgb_dir,
            args.tir_dir,
            split="val",
            val_split=args.val_split,
            img_size=args.img_size
        )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    psnr_vals = []
    filenames = []

    print(f"Evaluating {len(ds)} pairs...")

    with torch.no_grad():
        for tir, rgb in loader:
            tir = tir.to(device)
            rgb = rgb.to(device)

            fake = G(tir)

            # Full-image PSNR for this batch
            batch_psnr = psnr(fake, rgb)
            psnr_vals.append(batch_psnr)
            filenames.extend(ds.filenames[
                len(filenames) : len(filenames) + tir.size(0)
            ])

    import numpy as np
    psnr_arr = np.array(psnr_vals)
    mean_psnr = psnr_arr.mean()
    std_psnr = psnr_arr.std()

    print(f"Mean PSNR (batch-wise): {mean_psnr:.2f} dB")
    print(f"Std PSNR  (batch-wise): {std_psnr:.2f} dB")

    # For a per-file table, we approximate by assigning the batch PSNR to each image in the batch.
    # If you want exact per-image PSNR, we can change this to compute PSNR per image in the batch.
    per_file_psnr = []
    idx = 0
    for tir, rgb in loader:
        B = tir.size(0)
        with torch.no_grad():
            fake = G(tir.to(device))
        batch_psnr = psnr_full(fake, rgb.to(device))
        per_file_psnr.extend([batch_psnr] * B)
        idx += B

    # Save table: file_number, psnr
    def get_id_from_name(name: str) -> int:
        return int(Path(name).stem)

    numbers = [get_id_from_name(n) for n in filenames]
    tab = list(zip(numbers, per_file_psnr))

    out_txt = Path(args.out_dir) / "psnr_results.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt, "w") as f:
        f.write("file_number,psnr_db\n")
        for num, val in tab:
            f.write(f"{num},{val:.4f}\n")

    print(f"Per-file PSNR saved to: {out_txt}")


def get_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--test_root", type=str, default=None,
                   help="Optional separate test folder with same class1/class2 layout")
    p.add_argument("--rgb_dir", type=str, default="class1")
    p.add_argument("--tir_dir", type=str, default="class2")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--img_size", type=int, default=256)

    # Model
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to saved generator, e.g. checkpoints/G_final.pth")

    # Eval
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", type=str, default="eval_results")

    return p.parse_args()


if __name__ == "__main__":
    evaluate_psnr(get_args())