"""
train_pl.py — PyTorch Lightning training for Urban100 inpainting
================================================================
Models  : FNO | SNO | USNO | CascadeUSNO
Logging : TensorBoard  (scalars + full-image grids every val_every epochs)
Ckpts   : top-3 by val/rl2  +  last  (via ModelCheckpoint callback)

Usage:
    python3 train_pl.py \
        --img_dir  /workspace/data/SNO/Urban           \
        --data_dir /workspace/data/SNO                 \
        --out_dir  /workspace/project/SNO/results      \
        --ckpt_dir /workspace/project/SNO/checkpoints  \
        --H 128 --W 128 --hidden 64 --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from fno_urban           import FNO
from sno_urban           import SNO, build_shearlet_filters
from usno_urban          import USNO
from cascade_usno_urban  import CascadeUSNO
from urban_dataset import Urban100Dataset, Urban100FolderDataset


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = (pred - target).reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (d.norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean() / ((target ** 2).mean().clamp(min=1e-8))


# ─────────────────────────────────────────────────────────────────────────────
#  LightningModule
# ─────────────────────────────────────────────────────────────────────────────

class InpaintingLitModule(pl.LightningModule):
    """
    Wraps any of the four inpainting models (FNO/SNO/USNO/CascadeUSNO).
    Patch-based train/val/test steps; full-image eval handled by callback.
    """

    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        self.model = model
        self.cfg   = cfg
        # Save everything except the model object itself
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Training ──────────────────────────────────────────────────────────────
    def training_step(self, batch, _batch_idx):
        a, u  = batch
        pred  = self(a)
        loss  = mse_loss(pred, u)
        rl2   = rel_l2_batch(pred.detach(), u)
        self.log('train/loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('train/rl2',  rl2,  prog_bar=True,  on_step=False, on_epoch=True)
        return loss

    # ── Validation (patch-based, every epoch) ─────────────────────────────────
    def validation_step(self, batch, _batch_idx):
        a, u = batch
        pred = self(a)
        rl2  = rel_l2_batch(pred, u)
        self.log('val/rl2', rl2, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)

    # ── Test (patch-based) ────────────────────────────────────────────────────
    def test_step(self, batch, _batch_idx):
        a, u   = batch
        pred   = self(a)
        rl2    = rel_l2_batch(pred, u)
        self.log('test/rl2', rl2, on_step=False, on_epoch=True, sync_dist=True)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    def configure_optimizers(self):
        opt   = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.cfg['epochs'],
            eta_min=self.cfg['lr'] * 0.01)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}


# ─────────────────────────────────────────────────────────────────────────────
#  DataModule
# ─────────────────────────────────────────────────────────────────────────────

class Urban100DataModule(pl.LightningDataModule):
    """
    train : Urban100Dataset   (first 90 images, on-the-fly ff_mask, patches)
    val   : Urban100FolderDataset (urban100_val/, fixed masks, sliding patches)
    test  : Urban100FolderDataset (urban100_test/, fixed masks, sliding patches)
    """

    def __init__(self, img_dir: str, data_dir: str,
                 H: int = 128, W: int = 128,
                 batch_size: int = 16,
                 num_workers: int = 4):
        super().__init__()
        self.img_dir     = Path(img_dir)
        self.data_dir    = Path(data_dir)
        self.H, self.W   = H, W
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        exts        = {'.png', '.jpg', '.jpeg', '.bmp'}
        all_paths   = sorted({p for e in exts
                               for p in self.img_dir.rglob(f'*{e}')
                               if 'LR' not in str(p)
                               and 'bicubic' not in str(p)})
        train_paths = all_paths[:90]

        if stage in ('fit', None):
            self.train_ds = Urban100Dataset(
                train_paths, H=self.H, W=self.W,
                n_per_epoch=1)          # placeholder; updated below

            # Derive epoch length from the actual loaded images:
            # count non-overlapping H×W patches per image and sum.
            # Each image contributes floor((H_img - H) / H + 1) *
            #                           floor((W_img - W) / W + 1) patches.
            H, W = self.H, self.W
            n_per_epoch = sum(
                ((img.shape[0] - H) // H + 1) *
                ((img.shape[1] - W) // W + 1)
                for img in self.train_ds.images
            )
            self.train_ds.n_per_epoch = n_per_epoch
            print(f"  Epoch length: {n_per_epoch} patches "
                  f"({len(self.train_ds.images)} images × "
                  f"avg {n_per_epoch // len(self.train_ds.images)} patches/image)")
            self.val_ds   = Urban100FolderDataset(
                self.data_dir / 'urban100_val',
                H=self.H, W=self.W, stride=self.H)

        if stage in ('test', None):
            self.test_ds  = Urban100FolderDataset(
                self.data_dir / 'urban100_test',
                H=self.H, W=self.W, stride=self.H)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Full-image validation callback
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_window(H: int, W: int) -> np.ndarray:
    gy = np.exp(-((np.linspace(-1, 1, H)) ** 2) / 0.5)
    gx = np.exp(-((np.linspace(-1, 1, W)) ** 2) / 0.5)
    return np.outer(gy, gx).astype(np.float32)


@torch.no_grad()
def _patch_inference(model: nn.Module, masked: np.ndarray,
                     H: int, W: int, stride: int, device) -> np.ndarray:
    C, H_img, W_img = masked.shape
    win    = _gaussian_window(H, W)
    pad_h  = (H - H_img % H) % H
    pad_w  = (W - W_img % W) % W
    padded = np.pad(masked, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
    H_pad, W_pad = padded.shape[1], padded.shape[2]
    accum  = np.zeros((C, H_img, W_img), dtype=np.float64)
    count  = np.zeros((H_img, W_img), dtype=np.float64)
    model.eval()
    for y in range(0, H_pad - H + 1, stride):
        for x in range(0, W_pad - W + 1, stride):
            t    = torch.from_numpy(padded[:, y:y+H, x:x+W][None]).to(device)
            pred = model(t).cpu().numpy()[0]
            y1, y2 = min(y, H_img), min(y+H, H_img)
            x1, x2 = min(x, W_img), min(x+W, W_img)
            accum[:, y1:y2, x1:x2] += pred[:, :y2-y, :x2-x] * win[:y2-y, :x2-x]
            count[y1:y2, x1:x2]    += win[:y2-y, :x2-x]
    return (accum / np.maximum(count, 1e-8)).astype(np.float32)


def _load_png(path: Path) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1)


class FullImageValCallback(pl.Callback):
    """
    Every `every_n_epochs` epochs: run sliding-window inference on all full
    images in val_dir, log mean relL2 and image grids to TensorBoard.
    """

    def __init__(self, val_dir: Path, H: int, W: int,
                 every_n_epochs: int = 10, n_vis: int = 4):
        super().__init__()
        self.val_dir        = Path(val_dir)
        self.H, self.W      = H, W
        self.every_n_epochs = every_n_epochs
        self.n_vis          = n_vis
        self._gt_paths      = sorted((self.val_dir / 'gt').glob('*.png'))
        self._masked_paths  = sorted((self.val_dir / 'masked').glob('*.png'))

    def on_validation_epoch_end(self, trainer: pl.Trainer,
                                 pl_module: InpaintingLitModule) -> None:
        ep = trainer.current_epoch
        if ep % self.every_n_epochs != 0 and ep != trainer.max_epochs - 1:
            return

        device  = pl_module.device
        model   = pl_module.model
        stride  = self.H // 2
        scores  = []

        for i, (gp, mp) in enumerate(
                zip(self._gt_paths, self._masked_paths)):
            gt     = _load_png(gp)
            masked = _load_png(mp)
            pred   = _patch_inference(model, masked,
                                       self.H, self.W, stride, device)
            scores.append(
                float(np.sqrt(np.mean((pred - gt) ** 2)) /
                      (np.sqrt(np.mean(gt ** 2)) + 1e-8)))

            if i < self.n_vis and trainer.logger:
                row  = np.stack([
                    np.clip(gt   / 255.0, 0, 1),
                    np.clip(masked / 255.0, 0, 1),
                    np.clip(pred / 255.0, 0, 1),
                ])                                             # (3, 3, H, W)
                h_d  = min(256, gt.shape[1])
                w_d  = int(gt.shape[2] * h_d / gt.shape[1])
                row  = TF.interpolate(
                    torch.from_numpy(row),
                    size=(h_d, w_d), mode='bilinear', align_corners=False)
                trainer.logger.experiment.add_images(
                    f'val_full/img_{i:03d}', row, ep)

        mean_rl2 = float(np.mean(scores))
        pl_module.log('val/full_rl2', mean_rl2, prog_bar=False)
        if trainer.logger:
            trainer.logger.experiment.add_scalar(
                'val/full_rl2', mean_rl2, ep)
        print(f"\n  [full-image val  ep {ep:4d}]  "
              f"mean relL2 = {mean_rl2:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(key: str, cfg: dict, H: int, W: int) -> nn.Module:
    n_sh = build_shearlet_filters(H, W).shape[2]
    sz   = (H, W)
    C    = cfg['hidden']
    if key == 'fno':
        return FNO(in_channels=3, out_channels=3, hidden_channels=C,
                    n_sh=n_sh, n_blocks=cfg['n_blocks'])
    if key == 'sno':
        return SNO(in_channels=3, out_channels=3, hidden_channels=C,
                    n_blocks=cfg['n_blocks'], n_scales=cfg.get('n_scales'),
                    input_size=sz)
    if key == 'usno':
        return USNO(in_channels=3, out_channels=3, hidden_channels=C,
                     n_scales=cfg.get('n_scales'), n_layers=cfg['n_layers'],
                     input_size=sz)
    if key == 'cascade':
        return CascadeUSNO(in_channels=3, out_channels=3, hidden_channels=C,
                            n_scales=cfg.get('n_scales'), input_size=sz)
    raise ValueError(f"Unknown model key: {key}")


# ─────────────────────────────────────────────────────────────────────────────
#  Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    H, W = cfg['H'], cfg['W']

    # DataModule (shared across all models)
    dm = Urban100DataModule(
        img_dir     = cfg['img_dir'],
        data_dir    = cfg['data_dir'],
        H=H, W=W,
        batch_size  = cfg['batch_size'],
        num_workers = cfg['num_workers'])

    val_dir  = Path(cfg['data_dir']) / 'urban100_val'
    ckpt_dir = Path(cfg['ckpt_dir'])
    out_dir  = Path(cfg['out_dir'])

    for key in cfg['methods']:
        print(f"\n{'='*60}")
        print(f"  Training {key.upper()}   "
              f"{H}×{W}  hidden={cfg['hidden']}  epochs={cfg['epochs']}")
        print(f"{'='*60}")

        model      = build_model(key, cfg, H, W)
        lit_module = InpaintingLitModule(model, cfg)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        # ── Callbacks ─────────────────────────────────────────────────────────
        ckpt_cb = ModelCheckpoint(
            dirpath   = ckpt_dir / key,
            filename  = f'{key}-{{epoch:04d}}-{{val/rl2:.5f}}',
            monitor   = 'val/rl2',
            mode      = 'min',
            save_top_k= 3,
            save_last = True,
            verbose   = False)

        callbacks = [
            ckpt_cb,
            LearningRateMonitor(logging_interval='epoch'),
            FullImageValCallback(
                val_dir,
                H=H, W=W,
                every_n_epochs = cfg['val_every'],
                n_vis          = cfg['n_vis']),
        ]

        # ── Logger ─────────────────────────────────────────────────────────────
        logger = TensorBoardLogger(
            save_dir = str(out_dir),
            name     = 'tensorboard',
            version  = key)

        # ── Resume if checkpoint exists ────────────────────────────────────────
        ckpt_path = None
        last_ckpt  = ckpt_dir / key / 'last.ckpt'
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"  Resuming from {last_ckpt}")

        # ── Setup data (needed to know epoch length before Trainer is built) ───
        dm.setup('fit')
        steps_per_epoch = dm.train_ds.n_per_epoch // cfg['batch_size']

        # ── Trainer ────────────────────────────────────────────────────────────
        trainer = pl.Trainer(
            max_epochs            = cfg['epochs'],
            callbacks             = callbacks,
            logger                = logger,
            accelerator           = 'auto',
            devices               = 1,
            log_every_n_steps     = max(1, steps_per_epoch // 10),
            check_val_every_n_epoch = 5,
            num_sanity_val_steps  = 0,
            enable_progress_bar   = True,
            enable_model_summary  = True,
        )

        trainer.fit(lit_module, datamodule=dm, ckpt_path=ckpt_path)

        best = ckpt_cb.best_model_path
        score = ckpt_cb.best_model_score
        print(f"\n  [{key}] Best val/rl2 = {score:.5f}")
        print(f"  Best ckpt: {best}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Urban100 inpainting with PyTorch Lightning')
    p.add_argument('--img_dir',    default='/workspace/data/SNO/Urban')
    p.add_argument('--data_dir',   default='/workspace/data/SNO')
    p.add_argument('--out_dir',    default='/workspace/project/SNO/results')
    p.add_argument('--ckpt_dir',   default='/workspace/project/SNO/checkpoints')
    p.add_argument('--H',          type=int,   default=128)
    p.add_argument('--W',          type=int,   default=128)
    p.add_argument('--hidden',     type=int,   default=32)
    p.add_argument('--n_blocks',   type=int,   default=4)
    p.add_argument('--n_layers',   type=int,   default=5)
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--batch',      type=int,   default=16)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--val_every',  type=int,   default=10)
    p.add_argument('--n_vis',      type=int,   default=4)
    p.add_argument('--n_scales',   type=int,   default=None)
    p.add_argument('--methods', nargs='+', default=['fno','sno','usno','cascade'],
                   choices=['fno','sno','usno','cascade'])
    args = p.parse_args()

    cfg = dict(
        img_dir     = args.img_dir,
        data_dir    = args.data_dir,
        out_dir     = args.out_dir,
        ckpt_dir    = args.ckpt_dir,
        H           = args.H,
        W           = args.W,
        hidden      = args.hidden,
        n_blocks    = args.n_blocks,
        n_layers    = args.n_layers,
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        weight_decay= 1e-4,
        num_workers = args.num_workers,
        val_every   = args.val_every,
        n_vis       = args.n_vis,
        n_scales    = args.n_scales,
        methods     = args.methods,
    )
    train(cfg)