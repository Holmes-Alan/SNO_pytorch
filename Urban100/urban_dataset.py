"""
urban100_dataset.py
===================
PyTorch datasets for Urban100 inpainting.

Urban100Dataset        — training, on-the-fly ff_mask, first 90 images
Urban100FolderDataset  — val/test, reads pre-saved PNG files (full images),
                         extracts patches via deterministic sliding window
make_urban100_loaders  — convenience function for train_all.py
"""

import random
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
#  Free-form mask  (source: Generative Inpainting)
#  1 = missing,  0 = observed
# ─────────────────────────────────────────────────────────────────────────────

def _get_ff_mask(h: int, w: int) -> np.ndarray:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Training dataset  (on-the-fly masks, first 90 images)
# ─────────────────────────────────────────────────────────────────────────────

class Urban100Dataset(Dataset):
    """
    On-the-fly inpainting dataset for Urban100 training.

    For each sample:
      1. Pick a random image from the pre-loaded training images
      2. Generate a fresh ff_mask for the full image
      3. Apply mask: a_full = img × (1 − mask)
      4. Crop a random H×W patch with mask fraction in [min_frac, max_frac]
         and pixel std ≥ min_std  (retry up to max_tries times)
      5. Return (a_patch, u_patch) as (3, H, W) float32 tensors [0, 255]
    """

    def __init__(self, img_paths: list,
                 H: int = 128, W: int = 128,
                 n_per_epoch: int = 800,
                 min_frac: float = 0.10,
                 max_frac: float = 0.70,
                 min_std:  float = 4.0,
                 max_tries: int  = 20):
        self.H, self.W       = H, W
        self.n_per_epoch     = n_per_epoch
        self.min_frac        = min_frac
        self.max_frac        = max_frac
        self.min_std         = min_std
        self.max_tries       = max_tries

        print(f"Loading {len(img_paths)} training images into memory...")
        self.images = []
        for path in img_paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  Skipping: {path.name}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            if img.shape[0] >= H and img.shape[1] >= W:
                self.images.append(img)

        total_mb = sum(i.nbytes for i in self.images) // 1024 // 1024
        print(f"  Loaded {len(self.images)} images  (~{total_mb} MB)")
        if not self.images:
            raise RuntimeError("No valid training images loaded.")

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, _idx: int) -> tuple:
        for _ in range(self.max_tries * len(self.images)):
            img    = random.choice(self.images)          # (H_img, W_img, 3)
            H_img, W_img = img.shape[:2]

            mask   = _get_ff_mask(H_img, W_img)          # full-image mask
            a_full = img * (1 - mask[:, :, None])

            y = random.randint(0, H_img - self.H)
            x = random.randint(0, W_img - self.W)

            frac = mask[y:y+self.H, x:x+self.W].mean()
            if frac < self.min_frac or frac > self.max_frac:
                continue

            u_patch = img   [y:y+self.H, x:x+self.W]
            if u_patch.std() < self.min_std:
                continue

            a_patch = a_full[y:y+self.H, x:x+self.W]
            u = torch.from_numpy(u_patch.transpose(2, 0, 1).copy())
            a = torch.from_numpy(a_patch.transpose(2, 0, 1).copy())
            return a, u

        # Fallback: centre crop, no filter
        img    = self.images[0]
        H_img, W_img = img.shape[:2]
        y = (H_img - self.H) // 2
        x = (W_img - self.W) // 2
        mask   = _get_ff_mask(H_img, W_img)
        a_full = img * (1 - mask[:, :, None])
        u = torch.from_numpy(img   [y:y+self.H, x:x+self.W].transpose(2,0,1).copy())
        a = torch.from_numpy(a_full[y:y+self.H, x:x+self.W].transpose(2,0,1).copy())
        return a, u


# ─────────────────────────────────────────────────────────────────────────────
#  Val / Test dataset  (reads pre-saved full-image PNGs, extracts patches)
# ─────────────────────────────────────────────────────────────────────────────

class Urban100FolderDataset(Dataset):
    """
    Reads full-image PNG pairs from:
        folder/gt/XXXX.png      — original full image  (RGB uint8)
        folder/masked/XXXX.png  — masked full image    (RGB uint8)

    Extracts H×W patches via a deterministic sliding window so every
    evaluation run returns the same patches in the same order.

    Returns (a, u):
        a : (3, H, W) float32 [0,255]  masked patch
        u : (3, H, W) float32 [0,255]  original patch
    """

    def __init__(self, folder: Path,
                 H: int = 128, W: int = 128,
                 stride: int = 128):
        self.H, self.W = H, W
        gt_paths     = sorted((folder / 'gt').glob('*.png'))
        masked_paths = sorted((folder / 'masked').glob('*.png'))
        if not gt_paths:
            raise FileNotFoundError(f"No PNGs in {folder}/gt/")
        assert len(gt_paths) == len(masked_paths), \
            f"gt/masked count mismatch in {folder}"

        self.gt_paths     = gt_paths
        self.masked_paths = masked_paths

        # Pre-compute all (image_idx, y, x) patch positions
        self.index: list[tuple[int, int, int]] = []
        for img_idx, path in enumerate(gt_paths):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h_img, w_img = img.shape
            for y in range(0, h_img - H + 1, stride):
                for x in range(0, w_img - W + 1, stride):
                    self.index.append((img_idx, y, x))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple:
        img_idx, y, x = self.index[idx]

        def _crop(path):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            return torch.from_numpy(
                img[y:y+self.H, x:x+self.W].transpose(2, 0, 1).copy())

        u = _crop(self.gt_paths[img_idx])
        a = _crop(self.masked_paths[img_idx])
        return a, u


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience loader for train_all.py
# ─────────────────────────────────────────────────────────────────────────────

def make_urban100_loaders(img_dir: str, data_dir: str,
                           H: int = 128, W: int = 128,
                           batch_size: int = 4,
                           n_per_epoch: int = 800,
                           num_workers: int = 4):
    """
    Returns (train_loader, val_loader, test_loader).

    train_loader : Urban100Dataset     (first 90 images, on-the-fly ff_mask)
    val_loader   : Urban100FolderDataset  (urban100_val/,  fixed masks)
    test_loader  : Urban100FolderDataset  (urban100_test/, fixed masks)
    """
    img_dir  = Path(img_dir)
    data_dir = Path(data_dir)
    exts     = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths    = sorted({p for e in exts for p in img_dir.rglob(f'*{e}')
                       if 'LR' not in str(p) and 'bicubic' not in str(p)})

    train_ds = Urban100Dataset(paths[:90], H=H, W=W, n_per_epoch=n_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers,
                               pin_memory=True, drop_last=True)

    def _folder_loader(split, bs):
        folder = data_dir / f'urban100_{split}'
        if not folder.exists():
            raise FileNotFoundError(
                f"{folder} not found — run prepare_urban100.py first.")
        ds = Urban100FolderDataset(folder, H=H, W=W, stride=H)
        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=2, pin_memory=True)

    val_loader  = _folder_loader('val',  batch_size)
    test_loader = _folder_loader('test', batch_size)
    return train_loader, val_loader, test_loader