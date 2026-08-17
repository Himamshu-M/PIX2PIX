"""
Run a trained TIR -> RGB generator on new thermal images.

  python predict_tir2rgb.py --checkpoint checkpoints/G_final.pth --input tir_images --output rgb_out
  python predict_tir2rgb.py --checkpoint checkpoints/G_final.pth --input one_frame.png --output rgb_out

--input takes a single image or a folder. Output is written as PNG, one file
per input, named after the input.

The checkpoint carries its own config (input size, channel counts, the
normalization used in training), so nothing has to be passed in to match the
run that produced it. Bare state_dicts from older runs still load; the training
defaults are assumed and a warning is printed.

Requires: torch, torchvision, pillow
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from pix2pix_tir2rgb import GeneratorUNet

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULTS = {"in_channels": 1, "out_channels": 3, "img_size": 256,
            "norm_mean": 0.5, "norm_std": 0.5}


def load_generator(path, device):
    ck = torch.load(path, map_location=device)

    if isinstance(ck, dict) and "state_dict" in ck:
        cfg, state = {**DEFAULTS, **ck}, ck["state_dict"]
    else:
        print("Note: checkpoint holds only weights, no config. Assuming the "
              f"training defaults ({DEFAULTS['img_size']}px, 1 -> 3 channels).")
        cfg, state = dict(DEFAULTS), ck

    G = GeneratorUNet(cfg["in_channels"], cfg["out_channels"]).to(device)
    G.load_state_dict(state)
    G.eval()
    return G, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="e.g. checkpoints/G_final.pth")
    p.add_argument("--input", required=True, help="TIR image, or folder of TIR images")
    p.add_argument("--output", required=True, help="Folder to write RGB results into")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_size", choices=["orig", "model"], default="orig",
                   help="orig = resize back to each input's own dimensions; "
                        "model = leave at the model's square working size")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, cfg = load_generator(args.checkpoint, device)
    size = cfg["img_size"]
    print(f"Loaded {args.checkpoint} on {device} ({size}px input)")

    in_path = Path(args.input)
    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(f for f in in_path.iterdir()
                       if f.is_file() and f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found in {in_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((size, size), Image.BICUBIC),
        transforms.Grayscale(num_output_channels=cfg["in_channels"]),
        transforms.ToTensor(),
        transforms.Normalize([cfg["norm_mean"]], [cfg["norm_std"]]),
    ])

    print(f"{len(files)} images -> {out_dir}")
    for start in range(0, len(files), args.batch_size):
        chunk = files[start:start + args.batch_size]
        tensors, orig_sizes = [], []
        for f in chunk:
            im = Image.open(f).convert("L")
            orig_sizes.append(im.size)          # (width, height)
            tensors.append(tf(im))

        with torch.no_grad():
            out = G(torch.stack(tensors).to(device)).cpu()

        for f, img, (w, h) in zip(chunk, out, orig_sizes):
            if args.out_size == "orig":
                img = F.interpolate(img[None], size=(h, w), mode="bicubic",
                                    align_corners=False)[0]
            # Tanh output spans [-1, 1]; value_range maps and clamps it to [0, 1]
            save_image(img, out_dir / (f.stem + ".png"),
                       normalize=True, value_range=(-1, 1))

        print(f"  {min(start + args.batch_size, len(files))}/{len(files)}")

    print("Done.")


if __name__ == "__main__":
    main()