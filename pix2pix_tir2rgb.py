"""
Pix2Pix (conditional GAN) for TIR -> RGB image translation.

Architecture:
  Generator     : U-Net (encoder-decoder with skip connections)
  Discriminator : PatchGAN (70x70 receptive field)
  Loss          : Adversarial (BCE / LSGAN) + L1 pixel loss (lambda_L1 = 100)

Expected data layout (flat, auto-split into train/val internally):
  dataset/
    class1/   0001.jpg 0002.jpg ...   <- RGB images
    class2/   0001.jpg 0002.jpg ...   <- TIR images (same filenames, spatially aligned)

If your folders are named differently, pass --rgb_dir and --tir_dir.

Quality-check samples:
  By default the sample grids are drawn from the held-out val split. To watch a
  fixed set of pairs you picked yourself, put them in their own folder with the
  same class1/class2 layout and pass --sample_root. Keep that folder outside
  --data_root so those pairs are never trained on.

  sample_pairs/
    class1/   scene_a.jpg scene_b.jpg ...
    class2/   scene_a.jpg scene_b.jpg ...

Usage:
  python pix2pix_tir2rgb.py --data_root dataset --rgb_dir class1 --tir_dir class2 --epochs 200 --batch_size 4
  python pix2pix_tir2rgb.py --data_root dataset --sample_root sample_pairs --n_samples 6 --show

Requires: torch, torchvision, pillow  (matplotlib only if you want the grids displayed)
  pip install torch torchvision pillow matplotlib
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast, GradScaler

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class TIR2RGBDataset(Dataset):
    """Loads paired (TIR, RGB) images with matching filenames, 
    splits them, and applies perfectly aligned joint augmentations."""

    def __init__(self, root, rgb_dir="class1", tir_dir="class2",
                 split="train", val_split=0.1, img_size=256, seed=42):
        self.rgb_dir = Path(root) / rgb_dir
        self.tir_dir = Path(root) / tir_dir
        self.split = split  # Save split to check during __getitem__

        rgb_names = {f.name for f in self.rgb_dir.iterdir() if f.is_file()}
        tir_names = {f.name for f in self.tir_dir.iterdir() if f.is_file()}
        matched = sorted(rgb_names & tir_names)

        missing_rgb = tir_names - rgb_names
        missing_tir = rgb_names - tir_names
        if missing_rgb or missing_tir:
            print(f"Warning: {len(missing_rgb)} TIR files and {len(missing_tir)} "
                  f"RGB files have no matching pair and will be skipped.")
        if not matched:
            raise RuntimeError("No matching filenames found.")

        import random
        rng = random.Random(seed)
        shuffled = matched[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
        val_files = set(shuffled[:n_val])

        if split == "all":
            self.filenames = matched
        elif split == "train":
            self.filenames = [f for f in matched if f not in val_files]
        else:
            self.filenames = [f for f in matched if f in val_files]

        print(f"[{split}] {len(self.filenames)} paired images "
              f"(total matched pairs: {len(matched)})")

        self.tir_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        tir = Image.open(self.tir_dir / name).convert("L")
        rgb = Image.open(self.rgb_dir / name).convert("RGB")

        # Joint Augmentation: Apply same spatial transforms to both images
        if self.split == "train":
            import random
            if random.random() > 0.5:
                tir = TF.hflip(tir)
                rgb = TF.hflip(rgb)

        return self.tir_transform(tir), self.rgb_transform(rgb)


# --------------------------------------------------------------------------
# Generator: U-Net
# --------------------------------------------------------------------------
class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    """in_channels=1 (TIR grayscale) -> out_channels=3 (RGB)."""

    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# --------------------------------------------------------------------------
# Discriminator: PatchGAN
# --------------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    """Input is concat(condition, image) -> in_channels = tir_ch + rgb_ch."""

    def __init__(self, in_channels=4):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, cond_img, target_img):
        x = torch.cat((cond_img, target_img), 1)
        return self.model(x)


# --------------------------------------------------------------------------
# Quality-check samples
# --------------------------------------------------------------------------
def build_sample_batch(args, device):
    """One fixed batch of pairs, loaded once and reused for every sample grid.

    Uses --sample_root if given, otherwise the held-out val split. Picks are
    spread evenly across the set rather than taking the first N, because
    consecutive filenames in a TIR dataset are usually near-identical frames."""
    if args.sample_root:
        ds = TIR2RGBDataset(args.sample_root, args.rgb_dir, args.tir_dir,
                            split="all", img_size=args.img_size)
        source = args.sample_root
    else:
        ds = TIR2RGBDataset(args.data_root, args.rgb_dir, args.tir_dir,
                            split="val", val_split=args.val_split, img_size=args.img_size)
        source = "held-out val split"

    if len(ds) == 0:
        print("No pairs available for quality checks; sample grids disabled.")
        return None, None

    n = min(args.n_samples, len(ds))
    picks = [i * len(ds) // n for i in range(n)]
    loader = DataLoader(Subset(ds, picks), batch_size=n, shuffle=False)
    tir, rgb = next(iter(loader))

    print(f"Quality-check set: {n} pairs from {source}")
    for i in picks:
        print(f"    {ds.filenames[i]}")
    return tir.to(device), rgb.to(device)


def psnr(pred, target, patch_size=64):
    """
    Patch-based mean PSNR in dB for tensors in [-1, 1].
    Evaluates 5 patches (4 corners, 1 center) and uses the worst-case (max) MSE
    to ensure the images are only considered similar if ALL patches are similar.
    """
    B, C, H, W = pred.shape
    p = patch_size
    
    # 1. Define slices for the 5 patches: TL, TR, BL, BR, Center
    slices = [
        (slice(None), slice(None), slice(0, p), slice(0, p)),          # Top-Left
        (slice(None), slice(None), slice(0, p), slice(-p, None)),      # Top-Right
        (slice(None), slice(None), slice(-p, None), slice(0, p)),      # Bottom-Left
        (slice(None), slice(None), slice(-p, None), slice(-p, None)),  # Bottom-Right
        (slice(None), slice(None), slice(H//2 - p//2, H//2 + p//2), 
                                   slice(W//2 - p//2, W//2 + p//2))    # Center
    ]
    
    patch_mses = []
    for s in slices:
        pred_patch = pred[s]
        target_patch = target[s]
        # Calculate MSE for this specific patch
        mse = torch.mean((pred_patch - target_patch) ** 2, dim=[1, 2, 3])
        patch_mses.append(mse)
        
    # Stack MSEs -> shape (Batch, 5)
    patch_mses = torch.stack(patch_mses, dim=1)
    
    # 2. Take the maximum MSE across the 5 patches for each image
    # This guarantees that if even ONE patch is highly dissimilar, the score reflects it.
    worst_mse, _ = torch.max(patch_mses, dim=1)
    
    # Prevent division by zero just in case of a perfect match
    worst_mse = torch.clamp(worst_mse, min=1e-8)
    
    # 3. Compute PSNR (peak-to-peak is 2, so 2^2 = 4.0)
    batch_psnr = 10 * torch.log10(4.0 / worst_mse)
    
    return batch_psnr.mean().item()

def sample_and_report(G, tir, rgb, out_path, label, show):
    """Run the fixed pairs through G, save a 3-row grid, optionally display it.

    Rows: TIR input / generated RGB / ground truth. value_range is pinned to
    (-1, 1) so brightness is comparable between epochs instead of being
    rescaled per grid."""
    G.eval()
    with torch.no_grad():
        fake = G(tir)
    G.train()

    grid = torch.cat([tir.repeat(1, 3, 1, 1), fake, rgb], dim=0).cpu()
    save_image(grid, out_path, nrow=tir.size(0), normalize=True, value_range=(-1, 1))

    score = psnr(fake, rgb)
    print(f"  {label}: PSNR {score:.2f} dB  ->  {out_path}")

    if show:
        display_grid(out_path, f"{label}   PSNR {score:.2f} dB")
    return score


def display_grid(path, title, block=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (pip install matplotlib to display the grids)")
        return

    fig = plt.figure("pix2pix quality check", figsize=(13, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(plt.imread(path))
    ax.set_axis_off()
    ax.set_title(f"{title}\ntop: TIR input    middle: generated RGB    bottom: ground truth",
                 fontsize=10)
    fig.tight_layout()
    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.5)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    train_ds = TIR2RGBDataset(args.data_root, args.rgb_dir, args.tir_dir,
                               split="train", val_split=args.val_split, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    sample_tir, sample_rgb = build_sample_batch(args, device)
    has_samples = sample_tir is not None

    G = GeneratorUNet(in_channels=1, out_channels=3).to(device)
    D = PatchDiscriminator(in_channels=1 + 3).to(device)

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    lambda_l1 = args.lambda_l1

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Initialize Mixed Precision Scaler
    scaler = GradScaler()

    if has_samples:
        sample_and_report(G, sample_tir, sample_rgb,
                          f"{args.sample_dir}/epoch_0000.png", "epoch 0 (untrained)", args.show)

    for epoch in range(1, args.epochs + 1):
        for i, (tir, rgb) in enumerate(train_loader):
            tir, rgb = tir.to(device), rgb.to(device)

            with torch.no_grad():
                with autocast():
                    patch_out = D(tir, rgb)
            valid = torch.ones_like(patch_out, requires_grad=False)
            fake = torch.zeros_like(patch_out, requires_grad=False)

            # ---- Train Generator ----
            opt_G.zero_grad()
            with autocast():
                fake_rgb = G(tir)
                pred_fake = D(tir, fake_rgb)
                loss_gan = criterion_gan(pred_fake, valid)
                loss_l1 = criterion_l1(fake_rgb, rgb)
                loss_G = loss_gan + lambda_l1 * loss_l1
            
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            with autocast():
                pred_real = D(tir, rgb)
                loss_real = criterion_gan(pred_real, valid)
                # Detach fake_rgb so we don't backprop into G here
                pred_fake = D(tir, fake_rgb.detach())
                loss_fake = criterion_gan(pred_fake, fake)
                loss_D = 0.5 * (loss_real + loss_fake)
            
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            
            # Update scaler for next iteration
            scaler.update()

            if i % args.log_interval == 0:
                print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(train_loader)}] "
                      f"D_loss: {loss_D.item():.4f} G_loss: {loss_G.item():.4f} "
                      f"(gan {loss_gan.item():.4f}, l1 {loss_l1.item():.4f})")

        if has_samples and epoch % args.sample_interval == 0:
            sample_and_report(G, sample_tir, sample_rgb,
                              f"{args.sample_dir}/epoch_{epoch:04d}.png",
                              f"epoch {epoch}", args.show)

        if epoch % args.checkpoint_interval == 0:
            torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_epoch{epoch}.pth")
            torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_epoch{epoch}.pth")

    torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_final.pth")
    print("Training complete. Final generator saved.")

    if has_samples:
        score = sample_and_report(G, sample_tir, sample_rgb,
                                  f"{args.sample_dir}/final.png", "final", show=False)
        display_grid(f"{args.sample_dir}/final.png",
                     f"final   PSNR {score:.2f} dB", block=True)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    p.add_argument("--rgb_dir", type=str, default="class1", help="Folder name (under data_root) with RGB images")
    p.add_argument("--tir_dir", type=str, default="class2", help="Folder name (under data_root) with TIR images")
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of pairs held out for validation")
    p.add_argument("--sample_root", type=str, default=None,
                   help="Separate folder (same class1/class2 layout) of pairs used only for "
                        "quality checks. Defaults to the held-out val split.")
    p.add_argument("--n_samples", type=int, default=6, help="How many pairs to put in each sample grid")
    p.add_argument("--show", action="store_true",
                   help="Display each sample grid during training, not just at the end")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--sample_interval", type=int, default=1)
    p.add_argument("--checkpoint_interval", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--sample_dir", type=str, default="samples")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
