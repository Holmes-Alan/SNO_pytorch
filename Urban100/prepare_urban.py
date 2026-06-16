"""
prepare_urban100.py
===================
Generate VALIDATION and TEST image sets for Urban100 inpainting.

Each full image is processed as-is (no cropping).
One irregular mask is generated per image and applied to the full image.

Output structure:
    data/urban100_val/
        gt/       0000.png ... 0099.png   original full images  (RGB)
        masked/   0000.png ... 0099.png   masked full images    (RGB, holes=black)
        mask/     0000.png ... 0099.png   binary mask           (grayscale, 255=missing)

    data/urban100_test/
        gt/
        masked/
        mask/

Split:
    Training : first 90 images  →  Urban100Dataset  (on-the-fly masks)
    Val      : all 100 images   →  fixed mask seed 1000
    Test     : all 100 images   →  fixed mask seed 2000

Usage:
    python3 prepare_urban100.py \
        --img_dir  /scratch/.../Urban100/image_SRF_4 \
        --data_dir /scratch/.../SNO/data
"""

import argparse
import random
from pathlib import Path

import numpy as np
import cv2

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
#  Masks
# ─────────────────────────────────────────────────────────────────────────────

def get_ff_mask(h: int, w: int) -> np.ndarray:
    """Free-form brush-stroke mask.  1 = missing,  0 = observed."""
    mask  = np.zeros((h, w), dtype=np.float32)
    num_v = 15 + np.random.randint(9)
    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for _ in range(1 + np.random.randint(5)):
            angle   = 0.01 + np.random.randint(4)
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length  = 10 + np.random.randint(60)
            brush_w = 10 + np.random.randint(15)
            end_x   = int(start_x + length * np.sin(angle))
            end_y   = int(start_y + length * np.cos(angle))
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    return (mask > 0).astype(np.float32)


def read_rgb(path: Path) -> np.ndarray | None:
    """Return (H, W, 3) uint8 RGB, or None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_images(img_dir: Path) -> list:
    exts  = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths = [p for e in exts for p in img_dir.rglob(f'*{e}')
             if 'LR' not in str(p) and 'bicubic' not in str(p)]
    return sorted(set(paths))


# ─────────────────────────────────────────────────────────────────────────────
#  Build one split → save full images as PNG
# ─────────────────────────────────────────────────────────────────────────────

def build_split(img_paths: list, seed: int, out_dir: Path) -> None:
    """
    For each image:
        1. Generate one irregular mask for the full image
        2. Apply mask: masked = gt * (1 - mask)
        3. Save gt, masked, mask as PNG

    All images are saved at their original resolution.
    """
    np.random.seed(seed)
    random.seed(seed)

    gt_dir     = out_dir / 'gt';     gt_dir.mkdir(parents=True, exist_ok=True)
    masked_dir = out_dir / 'masked'; masked_dir.mkdir(exist_ok=True)
    mask_dir   = out_dir / 'mask';   mask_dir.mkdir(exist_ok=True)

    saved = 0; skipped = 0; fracs = []

    for path in img_paths:
        img = read_rgb(path)
        if img is None:
            print(f"  Skipping unreadable: {path.name}")
            skipped += 1
            continue

        H_img, W_img = img.shape[:2]

        # One irregular mask for the full image
        mask   = get_ff_mask(H_img, W_img)    # float32 {0,1}
        masked = np.clip(
            img.astype(np.float32) * (1 - mask[:, :, None]),
            0, 255
        ).astype(np.uint8)                            # (H, W, 3) uint8

        stem = f'{saved:04d}'

        # Save gt  (original full image, RGB → BGR for cv2)
        cv2.imwrite(str(gt_dir     / f'{stem}.png'),
                    cv2.cvtColor(img,    cv2.COLOR_RGB2BGR))

        # Save masked  (missing regions = black)
        cv2.imwrite(str(masked_dir / f'{stem}.png'),
                    cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

        # Save mask  (grayscale: 255 = missing, 0 = observed)
        cv2.imwrite(str(mask_dir   / f'{stem}.png'),
                    (mask * 255).astype(np.uint8))

        fracs.append(mask.mean())
        saved += 1

    print(f"  {saved} full images saved → {out_dir}")
    if fracs:
        print(f"  Mask coverage: mean={np.mean(fracs)*100:.1f}%  "
              f"range=[{np.min(fracs)*100:.1f}%, {np.max(fracs)*100:.1f}%]")
    if skipped:
        print(f"  ({skipped} images skipped)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir',  required=True,
                   help='Urban100 HR image folder (e.g. image_SRF_4)')
    p.add_argument('--data_dir', required=True,
                   help='Output root folder')
    args = p.parse_args()

    data_dir  = Path(args.data_dir)
    img_paths = find_images(Path(args.img_dir))
    print(f"Found {len(img_paths)} images in {args.img_dir}")
    if len(img_paths) < 10:
        raise FileNotFoundError(
            "Too few images found. Check --img_dir.\n"
            "Common sub-folders: image_SRF_4, HR")

    print(f"\nAll {len(img_paths)} images used for val and test "
          f"(full resolution, no cropping).")
    print(f"Training uses first 90 images via Urban100Dataset (on-the-fly).\n")

    print("Building val  (seed=1000)...")
    build_split(img_paths, seed=1000,
                out_dir=data_dir / 'urban100_val')

    print("\nBuilding test (seed=2000)...")
    build_split(img_paths, seed=2000,
                out_dir=data_dir / 'urban100_test')

    print(f"\nDone.")
    print(f"  Val  → {data_dir}/urban100_val/{{gt,masked,mask}}/")
    print(f"  Test → {data_dir}/urban100_test/{{gt,masked,mask}}/")
    print(f"\nTraining: run python3 train_all.py --dataset urban100 ...")


if __name__ == '__main__':
    main()