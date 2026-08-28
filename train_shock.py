"""
train_shock.py — FNO | SNO | USNO | CascadeUSNO on shock-bubble coarse->fine
============================================================================
Two tasks (see shock_dataset.py for the full rationale):

  --task same   129^2 -> 129^2   coarse sim -> fine sim at coarse nodes.
                Pure discretisation-error correction, no interpolation
                anywhere.  The cleanest architecture comparison; run this
                first, it is ~4x cheaper than 'sr'.

  --task sr     129^2 -> 257^2   bicubic(coarse) -> fine.
                Super-resolution on top of the correction.

Both predict the RESIDUAL to the input by default, so "do nothing" is an
explicit baseline (relL2 ~ 0.089) that is logged every validation epoch.

RUN THE FNO BANDWIDTH CONTROL
-----------------------------
Parameter-matched FNO gets k_max = round(sqrt(n_sh/2)) = 4, i.e. a 4x4 block
of Fourier modes.  It cannot represent a shock, so beating it says nothing
about shearlets.  Run both:

    python3 train_shock.py --task same --methods fno sno usno cascade
    python3 train_shock.py --task same --methods fno --k_max 32 --tag fno_wide

Only if SNO also beats the wide-band FNO is the result about the basis.

EXAMPLE
-------
    python3 train_shock.py \
        --root     /workspace/data/SNO/shock_bubble_sr \
        --out_dir  results/shock --ckpt_dir checkpoints/shock \
        --task same --hidden 32 --epochs 200 --batch 8

--out_dir and --ckpt_dir are relative to the working directory, so run this
from /workspace/project/SNO or pass absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import shock_common as SC
from shock_common import (build_model, describe_model, denormalise,
                          mse_loss, rel_l2, all_metrics, predict_frame)
from shock_dataset import (ShockDataset, DEFAULT_SPLIT, N_CHANNELS,
                           default_root, grid_size, load_or_compute_stats)


# ─────────────────────────────────────────────────────────────────────────────
#  LightningModule
# ─────────────────────────────────────────────────────────────────────────────

class ShockLitModule(pl.LightningModule):
    """
    Wraps any of the four operators.

    The network works in normalised units; every LOGGED metric is converted
    back to physical units first, so numbers are directly comparable across
    models, tasks, and against the do-nothing baseline.
    """

    def __init__(self, model: nn.Module, cfg: dict,
                 stats: Dict[str, List[float]]):
        super().__init__()
        self.model  = model
        self.cfg    = cfg
        self.stats  = stats
        self.target = cfg.get('target', 'residual')
        self.save_hyperparameters(ignore=['model'])
        self._val_rows: List[Dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── normalisation bookkeeping ────────────────────────────────────────────

    def _input_to_physical(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean, std, _ = SC.stats_tensors(self.stats, x_norm.device, x_norm.dtype)
        return x_norm * std + mean

    def _physical(self, out: torch.Tensor, x_norm: torch.Tensor):
        """Return (pred_phys, x_phys) for a normalised network output."""
        x_phys = self._input_to_physical(x_norm)
        return denormalise(out, x_phys, self.stats, self.target), x_phys

    # ── steps ────────────────────────────────────────────────────────────────

    def training_step(self, batch, _idx):
        x, y = batch
        pred = self(x)
        loss = mse_loss(pred, y)

        with torch.no_grad():
            p_phys, x_phys = self._physical(pred, x)
            t_phys, _      = self._physical(y,    x)
            self.log('train/loss', loss, on_step=False, on_epoch=True)
            self.log('train/rl2', rel_l2(p_phys, t_phys),
                     on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _idx):
        x, y           = batch
        pred           = self(x)
        p_phys, x_phys = self._physical(pred, x)
        t_phys, _      = self._physical(y,    x)

        m = all_metrics(p_phys, t_phys)
        m['baseline_rl2'] = rel_l2(x_phys, t_phys).item()   # do nothing
        self._val_rows.append(m)

    def on_validation_epoch_end(self):
        if not self._val_rows:
            return
        agg = {k: float(np.nanmean([r[k] for r in self._val_rows]))
               for k in self._val_rows[0]}
        self._val_rows.clear()

        for k, v in agg.items():
            self.log(f'val/{k}', v, prog_bar=(k == 'rl2'), sync_dist=True)
        # Duplicate without the slash: ModelCheckpoint interpolates the monitor
        # key into the filename, and a '/' there silently creates a directory.
        self.log('val_rl2', agg['rl2'], sync_dist=True)
        # Fraction of the do-nothing error that the operator removed.
        self.log('val/skill',
                 1.0 - agg['rl2'] / max(agg['baseline_rl2'], 1e-12),
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, _idx):
        x, y      = batch
        p_phys, _ = self._physical(self(x), x)
        t_phys, _ = self._physical(y,       x)
        for k, v in all_metrics(p_phys, t_phys).items():
            self.log(f'test/{k}', v, on_step=False, on_epoch=True, sync_dist=True)

    # ── optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg['lr'],
                                weight_decay=self.cfg['weight_decay'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg['epochs'], eta_min=self.cfg['lr'] * 0.01)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sch, 'interval': 'epoch'}}


# ─────────────────────────────────────────────────────────────────────────────
#  DataModule
# ─────────────────────────────────────────────────────────────────────────────

class ShockDataModule(pl.LightningDataModule):

    def __init__(self, cfg: dict, stats: Dict[str, List[float]]):
        super().__init__()
        self.cfg   = cfg
        self.stats = stats
        self.split = cfg['split']

    def _make(self, key: str, random_crop: bool) -> ShockDataset:
        c = self.cfg
        return ShockDataset(c['root'], self.split[key], task=c['task'],
                            target=c['target'], stats=self.stats,
                            patch=c['patch'],
                            crops_per_frame=c['crops_per_frame'],
                            random_crop=random_crop)

    def setup(self, stage: Optional[str] = None):
        if stage in ('fit', None):
            self.train_ds = self._make('train', True)
            self.val_ds   = self._make('val',   False)
        if stage in ('test', None):
            self.test_ds  = self._make('test',  False)

    def _loader(self, ds, shuffle: bool, drop_last: bool = False):
        return DataLoader(ds, batch_size=self.cfg['batch_size'],
                          shuffle=shuffle, drop_last=drop_last,
                          num_workers=self.cfg['num_workers'],
                          pin_memory=True,
                          persistent_workers=self.cfg['num_workers'] > 0)

    def train_dataloader(self):
        return self._loader(self.train_ds, True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)


# ─────────────────────────────────────────────────────────────────────────────
#  Full-frame validation
# ─────────────────────────────────────────────────────────────────────────────

class FullFrameValCallback(pl.Callback):
    """
    Periodically evaluate on WHOLE frames rather than patches.

    When training on patches the patch-level metric is optimistic: it never
    tests whether the operator is consistent across patch seams.  This runs
    the real sliding-window inference path used at test time.
    """

    def __init__(self, dataset: ShockDataset, cfg: dict,
                 every_n_epochs: int = 10, max_frames: int = 12):
        super().__init__()
        self.ds         = dataset
        self.cfg        = cfg
        self.every      = max(1, every_n_epochs)
        self.max_frames = max_frames

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        ep = trainer.current_epoch
        if ep % self.every != 0 and ep != trainer.max_epochs - 1:
            return

        n     = min(self.max_frames, self.ds.n_frames)
        rows  = []
        for i in range(n):
            x_phys, y_phys, _ = self.ds.full_frame(i)
            pred = predict_frame(pl_module.model, x_phys, self.ds.stats,
                                 target=self.cfg['target'],
                                 patch=self.cfg['patch'],
                                 stride=self.cfg['patch'] // 2 if self.cfg['patch'] else None,
                                 device=pl_module.device)
            p = torch.from_numpy(pred).unsqueeze(0)
            t = torch.from_numpy(y_phys).unsqueeze(0)
            m = all_metrics(p, t)
            m['baseline_rl2'] = rel_l2(torch.from_numpy(x_phys).unsqueeze(0), t).item()
            rows.append(m)

        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
        for k, v in agg.items():
            pl_module.log(f'val_full/{k}', v, sync_dist=True)
        print(f"\n  [full-frame val ep {ep:4d}]  relL2={agg['rl2']:.5f}  "
              f"highpass={agg['rl2_highpass']:.5f}  front={agg['rl2_front']:.5f}  "
              f"(baseline {agg['baseline_rl2']:.5f})")


class HistoryCSVCallback(pl.Callback):
    """Mirror the logged scalars into a CSV, so plots need no TensorBoard."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.rows: List[Dict[str, float]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        row = {'epoch': trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        self.rows.append(row)

        keys = sorted({k for r in self.rows for k in r})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

class _Absent:
    def __repr__(self): return '<absent>'


_ABSENT = _Absent()


def train(cfg: dict) -> None:
    root     = Path(cfg['root'])
    out_dir  = Path(cfg['out_dir']);  out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg['ckpt_dir']); ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_grid = grid_size(cfg['task'])
    n_in   = cfg['patch'] or n_grid
    cfg['n_grid'] = n_grid

    stats = load_or_compute_stats(root, cfg['split']['train'], cfg['task'],
                                  out_dir / f"stats_{cfg['task']}.json")
    with open(out_dir / f"config_{cfg['task']}{cfg['tag']}.json", 'w') as f:
        json.dump({k: v for k, v in cfg.items() if k != 'split'} |
                  {'split': cfg['split']}, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  task={cfg['task']}  grid={n_grid}²  model input={n_in}²  "
          f"pad={cfg['pad']}  target={cfg['target']}")
    print(f"  train={cfg['split']['train']}")
    print(f"  val  ={cfg['split']['val']}   test={cfg['split']['test']}")
    print(f"{'='*66}")

    dm = ShockDataModule(cfg, stats)
    dm.setup('fit')

    for key in cfg['methods']:
        name = f"{key}{cfg['tag']}"
        # Per-method copy with defaults resolved, so the checkpoint records the
        # architecture that was actually built rather than a pile of Nones.
        mcfg = SC.resolve_cfg(key, cfg)
        print(f"\n{'─'*66}\n  {SC.LABELS.get(key, key)}   {describe_model(key, mcfg, n_in)}\n{'─'*66}")

        model = build_model(key, mcfg, n_in)
        lit   = ShockLitModule(model, mcfg, stats)
        print(f"  parameters: {model.param_count():,}")

        ckpt_cb = ModelCheckpoint(
            dirpath    = ckpt_dir / name,
            filename   = f'{name}-{{epoch:04d}}-{{val_rl2:.5f}}',
            monitor    = 'val_rl2',
            mode       = 'min',
            save_top_k = 3,
            save_last  = True,
            auto_insert_metric_name = False)

        callbacks = [
            ckpt_cb,
            LearningRateMonitor(logging_interval='epoch'),
            HistoryCSVCallback(out_dir / f'{name}_history.csv'),
        ]
        if cfg['full_val_every'] > 0:
            callbacks.append(FullFrameValCallback(
                dm.val_ds, mcfg, every_n_epochs=cfg['full_val_every'],
                max_frames=cfg['full_val_frames']))

        trainer = pl.Trainer(
            max_epochs              = cfg['epochs'],
            callbacks               = callbacks,
            logger                  = TensorBoardLogger(save_dir=str(out_dir),
                                                        name='tensorboard',
                                                        version=name),
            accelerator             = 'auto',
            devices                 = 1,
            check_val_every_n_epoch = cfg['val_every'],
            num_sanity_val_steps    = 0,
            gradient_clip_val       = 1.0,
            enable_progress_bar     = True,
            enable_model_summary    = False)

        last   = ckpt_dir / name / 'last.ckpt'
        resume = str(last) if last.exists() else None
        if resume:
            # A stale last.ckpt from a DIFFERENT task is the trap here: 'same'
            # and 'sr' share checkpoints/shock/<method>, so an sr run happily
            # picks up a 129²-trained model.  FNO is resolution-independent
            # and would silently resume at the old epoch count; SNO/USNO/
            # Cascade die on a Psi shape mismatch.  Refuse both.
            prev = ((torch.load(resume, map_location='cpu',
                                weights_only=False).get('hyper_parameters')
                     or {}).get('cfg') or {})
            # Must include the METHOD-SPECIFIC keys, and must treat a key
            # that is ABSENT from the checkpoint as a mismatch.  warp_energy is
            # the motivating case: it changes what ShearletMix computes while
            # leaving every state_dict key and shape untouched, so a checkpoint
            # written before the flag existed loads cleanly and then trains on
            # silently different semantics.  Absent-vs-set has to count.
            differs = []
            for k in ('task', 'patch', 'pad', 'hidden', 'n_blocks', 'n_layers',
                      'n_scales', 'k_max', 'lifting_dim', 'n_levels',
                      'num_heads', 'groups', 'warp_guided', 'max_disp',
                      'warp_energy', 'sno_at', 'sno_channels'):
                a, b = prev.get(k, _ABSENT), mcfg.get(k, _ABSENT)
                if a == b or (a in (None, _ABSENT) and b in (None, _ABSENT)):
                    continue          # equal, or unset on both sides
                differs.append(k)
            if differs:
                raise SystemExit(
                    f"\n  {last} was trained with a different configuration:\n"
                    + "".join(f"    {k}: checkpoint="
                                f"{prev.get(k, _ABSENT)!r}  now={mcfg.get(k, _ABSENT)!r}\n"
                              for k in differs)
                    + "  Resuming it would splice two runs together.  Delete\n"
                      f"  {last.parent} to retrain from scratch, give this run\n"
                      "  its own directory with --tag, or point --ckpt_dir\n"
                      "  elsewhere.  '<absent>' means the checkpoint predates\n"
                      "  that option and would silently inherit its new default.")
            print(f"  resuming from {last}")

        trainer.fit(lit, datamodule=dm, ckpt_path=resume)
        print(f"\n  [{name}] best val_rl2 = {ckpt_cb.best_model_score}")
        print(f"  best ckpt: {ckpt_cb.best_model_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_cfg(args) -> dict:
    split = dict(DEFAULT_SPLIT)
    if args.split_json:
        with open(args.split_json) as f:
            split = json.load(f)
    return dict(
        root            = args.root,
        out_dir         = args.out_dir,
        ckpt_dir        = args.ckpt_dir,
        task            = args.task,
        target          = args.target,
        split           = split,
        patch           = args.patch,
        pad             = args.pad,
        crops_per_frame = args.crops_per_frame,
        hidden          = args.hidden,
        n_blocks        = args.n_blocks,
        n_layers        = args.n_layers,
        n_scales        = args.n_scales,
        k_max           = args.k_max,
        lifting_dim     = args.lifting_dim,
        n_levels        = args.n_levels,
        num_heads       = args.num_heads,
        warp_energy     = args.warp_energy,
        groups          = args.groups,
        in_channels     = N_CHANNELS,
        out_channels    = N_CHANNELS,
        epochs          = args.epochs,
        batch_size      = args.batch,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        num_workers     = args.num_workers,
        val_every       = args.val_every,
        full_val_every  = args.full_val_every,
        full_val_frames = args.full_val_frames,
        methods         = args.methods,
        tag             = f'_{args.tag}' if args.tag else '',
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',     default=default_root())
    p.add_argument('--out_dir',  default='results/shock')
    p.add_argument('--ckpt_dir', default='checkpoints/shock')
    p.add_argument('--task',     default='same', choices=['same', 'sr'])
    p.add_argument('--target',   default='residual', choices=['residual', 'direct'])
    p.add_argument('--split_json', default=None,
                   help='JSON with {"train":[...],"val":[...],"test":[...]}')
    p.add_argument('--patch',   type=int, default=0,
                   help='0 = whole frame. Use 128 for --task sr (257² will OOM).')
    p.add_argument('--pad',     type=int, default=16,
                   help='reflect padding to hide the periodic seam; 0 disables')
    p.add_argument('--crops_per_frame', type=int, default=4)
    p.add_argument('--hidden',   type=int, default=32)
    p.add_argument('--n_blocks', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--n_scales', type=int, default=None)
    p.add_argument('--k_max',    type=int, default=None,
                   help='FNO mode cutoff. None = param-matched to SNO (k=4). '
                        'Set 32 for the bandwidth control.')
    p.add_argument('--n_levels',    type=int, default=None,
                   help='Flower U-Net depth; padded grid must be divisible '
                        'by 2**(n_levels-1)')
    p.add_argument('--lifting_dim', type=int, default=None,
                   help="Flower width. None = param-matched to SNO (28).")
    p.add_argument('--warp_energy', default=None, choices=['scale', 'subband'],
                   help="warpsno flow-head conditioning: 'scale' (default, "
                        "J+1 bandpass filters) or 'subband' (exact, ~7x cost)")
    p.add_argument('--num_heads',   type=int, default=None,
                   help='warp heads for flower / warpsno')
    p.add_argument('--groups',      type=int, default=None,
                   help='GroupNorm groups for flower')
    p.add_argument('--epochs',   type=int,   default=200)
    p.add_argument('--batch',    type=int,   default=8)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--val_every',       type=int, default=5)
    p.add_argument('--full_val_every',  type=int, default=10,
                   help='0 disables full-frame validation')
    p.add_argument('--full_val_frames', type=int, default=12)
    p.add_argument('--tag', default='',
                   help='suffix for ckpt/log dirs, e.g. --tag fno_wide')
    p.add_argument('--methods', nargs='+', default=list(SC.METHODS),
                   choices=list(SC.METHODS))
    args = p.parse_args()

    if args.task == 'sr' and args.patch == 0:
        print("WARNING: --task sr at full 257² needs ~10+ GB per block with "
              "autograd.\n         Use --patch 128 unless you know the model fits.")

    train(build_cfg(args))
