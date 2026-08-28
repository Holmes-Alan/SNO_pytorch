"""
shock_dataset.py — Shock-bubble coarse/fine dataset for operator learning
==========================================================================
Data: /raid/.../shock_bubble_sr — 20 Latin-hypercube samples of the
compressible Euler shock-bubble interaction, each simulated TWICE from
identical initial data:

    coarse   -r 5  →  129×129 nodes
    fine     -r 6  →  257×257 nodes      (257 = 2·129 − 1)

Because 257 = 2·129 − 1 the coarse nodes are *exactly* the even-indexed
fine nodes, so `fine[:, ::2, ::2]` lives on the coarse grid with no
interpolation whatsoever.  At t=0 the two runs agree to machine zero; the
divergence at later times is pure discretisation error, and that is the
signal we ask the operator to learn.

TWO TASKS
---------
task='same'  129² → 129²      input  = coarse
                              target = fine[:, ::2, ::2]
    Pure discretisation-error correction.  No resolution change, no
    interpolation anywhere in the pipeline — the cleanest possible
    comparison between operator architectures.

task='sr'    129² → 257²      input  = bicubic(coarse, 257²)
                              target = fine
    Super-resolution on top of the correction.  The model predicts the
    RESIDUAL to the bicubic upsample, so bicubic is an explicit,
    free baseline (relL2 ≈ 0.076–0.104 measured over the test split).

WHY NOT A 65² LEVEL
-------------------
Decimating to 65² (fine[::4,::4] or coarse[::2,::2]) would put both sides
of the map inside the SAME simulation, leaving only interpolation error
(≈1–3%) and no discretisation error at all.  A real 65² level would need
the solver re-run at `-r 4`.

CHANNELS
--------
Conserved variables, MFEM ordering, indexed [channel, y, x]:
    0 rho     1 rho*u     2 rho*v     3 E
Channel magnitudes differ by orders of magnitude and the momenta are
near-zero over most of the domain, so per-channel standardisation is
mandatory — otherwise rho and E dominate every gradient.

SPLITS
------
Split by SAMPLE, never by snapshot: the 24 snapshots of one sample share
an initial condition.  Some samples are near-duplicates of each other
(s07↔s16 at relL2 0.093; also (04,10), (03,15), (14,19)) — the default
split keeps every such pair on the same side.  Run

    python3 shock_dataset.py --check_split

to re-verify leakage for any split you choose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Data root.  Training runs inside Docker, where the dataset is mounted at
# /workspace/data/SNO; the same volume is /raid/data/zhisong_liu/SNO on the
# host.  Both are probed so the self-tests work on either side of the mount.
# Override with $SHOCK_DATA_ROOT or --root.
ROOT_CANDIDATES = (
    '/workspace/data/SNO/shock_bubble_sr',
    '/raid/data/zhisong_liu/SNO/shock_bubble_sr',
)


def default_root() -> str:
    """Container path first, host path as fallback, env var wins over both."""
    env = os.environ.get('SHOCK_DATA_ROOT')
    if env:
        return env
    for candidate in ROOT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return ROOT_CANDIDATES[0]


CHANNELS   = ('rho', 'rho_u', 'rho_v', 'E')
N_CHANNELS = 4
N_COARSE   = 129
N_FINE     = 257
N_SNAPSHOT = 24
N_SAMPLE   = 20

# Sample-level split.  Near-duplicate pairs (07,16) (04,10) (03,15) (14,19)
# all fall inside 'train'; val/test samples are the most isolated ones.
DEFAULT_SPLIT: Dict[str, List[int]] = {
    'train': [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 15, 16, 17, 19],
    'val':   [5, 11, 18],
    'test':  [2, 8, 12],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Raw I/O
# ─────────────────────────────────────────────────────────────────────────────

def sample_dir(root: Path, s: int) -> Path:
    return Path(root) / f'sample_{s:02d}'


def load_field(root: Path, s: int, k: int, res: str) -> np.ndarray:
    """Load one snapshot.  res ∈ {'coarse','fine'}.  Returns (4, N, N) float64."""
    return np.load(sample_dir(root, s) / res / 'fields' / f't{k:04d}.npy')


def load_times(root: Path, s: int, res: str = 'fine') -> List[float]:
    with open(sample_dir(root, s) / res / 'times.json') as f:
        return json.load(f)['times']


def upsample_bicubic(coarse: np.ndarray, size: int = N_FINE) -> np.ndarray:
    """
    129² → 257² bicubic.

    align_corners=True is REQUIRED, not cosmetic: it maps corner node to
    corner node, so coarse node j lands exactly on fine node 2j, matching
    the grid relationship the dataset is built on.
    """
    t   = torch.from_numpy(np.ascontiguousarray(coarse)).float().unsqueeze(0)
    out = F.interpolate(t, size=(size, size), mode='bicubic',
                        align_corners=True)
    return out.squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(root: Path, sample_ids: Sequence[int],
                  task: str = 'same',
                  drop_t0: Optional[bool] = None) -> Dict[str, List[float]]:
    """
    Per-channel statistics over the TRAINING samples only.

    Returns
    -------
    {'mean': [4], 'std': [4], 'res_std': [4]}
        mean/std    : of the input field   → standardises the network input
        res_std     : of the target residual (target − input)
                      → puts the regression target at unit scale

    Both are plain per-channel scalars; no spatial structure is removed.
    """
    n     = 0
    s1    = np.zeros(N_CHANNELS, dtype=np.float64)
    s2    = np.zeros(N_CHANNELS, dtype=np.float64)
    r1    = np.zeros(N_CHANNELS, dtype=np.float64)
    r2    = np.zeros(N_CHANNELS, dtype=np.float64)

    for s in sample_ids:
        for k in default_snapshots(task, drop_t0):
            x, y = build_pair(root, s, k, task)
            npix = x.shape[1] * x.shape[2]
            s1  += x.reshape(N_CHANNELS, -1).sum(axis=1)
            s2  += (x.reshape(N_CHANNELS, -1) ** 2).sum(axis=1)
            r    = (y - x).reshape(N_CHANNELS, -1)
            r1  += r.sum(axis=1)
            r2  += (r ** 2).sum(axis=1)
            n   += npix

    mean    = s1 / n
    var     = np.maximum(s2 / n - mean ** 2, 0.0)
    std     = np.sqrt(var)
    res_var = np.maximum(r2 / n - (r1 / n) ** 2, 0.0)
    res_std = np.sqrt(res_var)

    # Guard against a degenerate channel (should not happen on this data)
    std     = np.where(std     < 1e-8, 1.0, std)
    res_std = np.where(res_std < 1e-8, 1.0, res_std)

    return {'mean': mean.tolist(), 'std': std.tolist(),
            'res_std': res_std.tolist()}


def load_or_compute_stats(root: Path, sample_ids: Sequence[int],
                          task: str, cache: Optional[Path] = None,
                          drop_t0: Optional[bool] = None
                          ) -> Dict[str, List[float]]:
    if cache is not None and Path(cache).exists():
        with open(cache) as f:
            payload = json.load(f)
        if (payload.get('task') == task and
                payload.get('sample_ids') == list(sample_ids)):
            return {k: payload[k] for k in ('mean', 'std', 'res_std')}
    stats = compute_stats(root, sample_ids, task, drop_t0)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, 'w') as f:
            json.dump({'task': task, 'sample_ids': list(sample_ids), **stats},
                      f, indent=2)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Pair construction
# ─────────────────────────────────────────────────────────────────────────────

def build_pair(root: Path, s: int, k: int, task: str
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (input, target) as float32 (4, N, N) on a common grid.

    task='same' : (coarse 129², fine[:, ::2, ::2] 129²)
    task='sr'   : (bicubic(coarse) 257², fine 257²)
    """
    lo = load_field(root, s, k, 'coarse')
    hi = load_field(root, s, k, 'fine')
    if task == 'same':
        return lo.astype(np.float32), np.ascontiguousarray(
            hi[:, ::2, ::2]).astype(np.float32)
    if task == 'sr':
        return upsample_bicubic(lo).astype(np.float32), hi.astype(np.float32)
    raise ValueError(f"task must be 'same' or 'sr', got {task!r}")


def grid_size(task: str) -> int:
    return N_COARSE if task == 'same' else N_FINE


def default_snapshots(task: str, drop_t0: Optional[bool] = None) -> List[int]:
    """
    Which snapshot indices to use.

    For task='same' the t=0 frame is DEGENERATE.  Both runs start from the
    same analytic initial condition sampled at the same nodes, so
    coarse == fine[:, ::2, ::2] to machine zero and the residual target is
    identically zero.  Keeping it would hand the model 1/24 of the dataset
    as a free identity example and bias every reported metric optimistically.
    It is dropped by default.

    For task='sr' the t=0 frame is NOT degenerate — bicubic upsampling of the
    sharp initial interface is genuinely lossy (relL2 ~ 0.062) — so it is kept.
    """
    if drop_t0 is None:
        drop_t0 = (task == 'same')
    return list(range(1, N_SNAPSHOT)) if drop_t0 else list(range(N_SNAPSHOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ShockDataset(Dataset):
    """
    Coarse→fine pairs, held in memory as float32.

    Every item is
        x : (4, h, w)  standardised input      (x_phys − mean) / std
        y : (4, h, w)  standardised target     depends on `target`:
                         'residual' → (y_phys − x_phys) / res_std
                         'direct'   → (y_phys − mean)   / std

    `patch` controls cropping:
        patch = 0     → whole frame, no cropping
        patch = P > 0 → P×P crops.  random positions when random_crop=True
                        (training), otherwise a deterministic stride-P grid
                        (validation / test), so eval is reproducible.

    Crops never wrap the domain, so a patch is never cut across the
    periodic seam that FFT-based operators would otherwise see at the
    reflecting walls.  See the note in `train_shock.py` about `pad`.
    """

    def __init__(self, root: Path, sample_ids: Sequence[int],
                 task: str = 'same',
                 target: str = 'residual',
                 stats: Optional[Dict[str, List[float]]] = None,
                 patch: int = 0,
                 crops_per_frame: int = 4,
                 random_crop: bool = True,
                 snapshots: Optional[Sequence[int]] = None,
                 drop_t0: Optional[bool] = None,
                 seed: int = 0,
                 verbose: bool = True):
        super().__init__()
        if target not in ('residual', 'direct'):
            raise ValueError(f"target must be 'residual' or 'direct', got {target!r}")

        self.root        = Path(root)
        self.sample_ids  = list(sample_ids)
        self.task        = task
        self.target      = target
        self.patch       = int(patch)
        self.random_crop = bool(random_crop)
        self.snapshots   = (default_snapshots(task, drop_t0)
                            if snapshots is None else list(snapshots))
        self.rng         = np.random.default_rng(seed)

        if stats is None:
            stats = compute_stats(self.root, self.sample_ids, task, drop_t0)
        self.stats = stats
        # Broadcastable (4, 1, 1) views
        self.mean    = np.asarray(stats['mean'],    dtype=np.float32)[:, None, None]
        self.std     = np.asarray(stats['std'],     dtype=np.float32)[:, None, None]
        self.res_std = np.asarray(stats['res_std'], dtype=np.float32)[:, None, None]

        # ── Load every frame into RAM ─────────────────────────────────────────
        if verbose:
            print(f"  Loading {len(self.sample_ids)} samples × "
                  f"{len(self.snapshots)} snapshots  (task={task})...")
        self.frames: List[Tuple[np.ndarray, np.ndarray]] = []
        self.frame_id: List[Tuple[int, int]] = []
        for s in self.sample_ids:
            for k in self.snapshots:
                self.frames.append(build_pair(self.root, s, k, task))
                self.frame_id.append((s, k))

        self.N = self.frames[0][0].shape[-1]
        if self.patch and self.patch > self.N:
            raise ValueError(f"patch={self.patch} exceeds grid size {self.N}")

        # ── Build the item index ──────────────────────────────────────────────
        # random: `crops_per_frame` items per frame, positions drawn per __getitem__
        # grid  : one item per (frame, y, x) on a stride-`patch` lattice
        self.index: List[Tuple[int, int, int]] = []
        if self.patch == 0:
            self.index = [(i, 0, 0) for i in range(len(self.frames))]
        elif self.random_crop:
            for i in range(len(self.frames)):
                self.index.extend((i, -1, -1) for _ in range(crops_per_frame))
        else:
            stride = self.patch
            pos    = list(range(0, self.N - self.patch + 1, stride))
            # Always include the far edge so the domain is fully covered
            if pos[-1] != self.N - self.patch:
                pos.append(self.N - self.patch)
            for i in range(len(self.frames)):
                for y in pos:
                    for x in pos:
                        self.index.append((i, y, x))

        if verbose:
            mb = sum(a.nbytes + b.nbytes for a, b in self.frames) // 1024 // 1024
            print(f"  {len(self.frames)} frames at {self.N}²  (~{mb} MB), "
                  f"{len(self.index)} items/epoch")

    # ── torch.utils.data.Dataset ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i, y, x = self.index[idx]
        xf, yf  = self.frames[i]

        if self.patch:
            if y < 0:
                hi = self.N - self.patch
                y  = int(self.rng.integers(0, hi + 1))
                x  = int(self.rng.integers(0, hi + 1))
            xf = xf[:, y:y + self.patch, x:x + self.patch]
            yf = yf[:, y:y + self.patch, x:x + self.patch]

        x_n = (xf - self.mean) / self.std
        if self.target == 'residual':
            y_n = (yf - xf) / self.res_std
        else:
            y_n = (yf - self.mean) / self.std

        return (torch.from_numpy(np.ascontiguousarray(x_n)),
                torch.from_numpy(np.ascontiguousarray(y_n)))

    # ── Denormalisation, used by the evaluators ──────────────────────────────

    def to_physical(self, out: torch.Tensor, x_phys: torch.Tensor
                    ) -> torch.Tensor:
        """
        Map a network output back to physical units.

        out    : (B, 4, h, w) network output, standardised
        x_phys : (B, 4, h, w) the physical input field the patch came from
        """
        dev = out.device
        if self.target == 'residual':
            rs = torch.as_tensor(self.res_std, dtype=out.dtype, device=dev)
            return x_phys + out * rs
        mu = torch.as_tensor(self.mean, dtype=out.dtype, device=dev)
        sd = torch.as_tensor(self.std,  dtype=out.dtype, device=dev)
        return mu + out * sd

    def full_frame(self, i: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Physical (input, target, (sample, snapshot)) for frame i — for eval."""
        xf, yf = self.frames[i]
        return xf, yf, self.frame_id[i]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


# ─────────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────────

def make_shock_loaders(root: Path,
                       task: str = 'same',
                       target: str = 'residual',
                       split: Optional[Dict[str, List[int]]] = None,
                       patch: int = 0,
                       crops_per_frame: int = 4,
                       batch_size: int = 8,
                       num_workers: int = 4,
                       stats_cache: Optional[Path] = None,
                       drop_t0: Optional[bool] = None,
                       verbose: bool = True):
    """
    Returns (train_loader, val_loader, test_loader, stats).

    Statistics come from the TRAIN split only and are shared by all three,
    which is the only leak-free choice.
    """
    split = split or DEFAULT_SPLIT
    root  = Path(root)
    stats = load_or_compute_stats(root, split['train'], task, stats_cache,
                                  drop_t0)

    def _ds(ids, random_crop):
        return ShockDataset(root, ids, task=task, target=target, stats=stats,
                            patch=patch, crops_per_frame=crops_per_frame,
                            random_crop=random_crop, drop_t0=drop_t0,
                            verbose=verbose)

    train_ds = _ds(split['train'], True)
    val_ds   = _ds(split['val'],   False)
    test_ds  = _ds(split['test'],  False)

    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              **common)
    return train_loader, val_loader, test_loader, stats


# ─────────────────────────────────────────────────────────────────────────────
#  Split leakage check
# ─────────────────────────────────────────────────────────────────────────────

def check_split(root: Path, split: Optional[Dict[str, List[int]]] = None,
                snapshot: int = N_SNAPSHOT - 1) -> None:
    """
    Print, for every val/test sample, its relL2 distance to the nearest
    TRAIN sample.  A small distance means that held-out sample is nearly a
    duplicate of something the model trained on, and its score is optimistic.
    """
    split = split or DEFAULT_SPLIT
    root  = Path(root)

    def rho(s):
        return load_field(root, s, snapshot, 'fine')[0]

    tr = {s: rho(s) for s in split['train']}

    def rl2(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    print(f"Split leakage check (fine rho, t index {snapshot})")
    print(f"  train: {split['train']}")
    for name in ('val', 'test'):
        print(f"  {name}:")
        for s in split[name]:
            f = rho(s)
            d = sorted((rl2(f, g), t) for t, g in tr.items())
            flag = '  <-- CLOSE, possible leakage' if d[0][0] < 0.12 else ''
            print(f"    s{s:02d}  nearest train = s{d[0][1]:02d} "
                  f"at relL2 {d[0][0]:.4f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='shock_bubble_sr dataset self-test')
    p.add_argument('--root', default=default_root())
    p.add_argument('--task', default='same', choices=['same', 'sr'])
    p.add_argument('--patch', type=int, default=0)
    p.add_argument('--check_split', action='store_true')
    args = p.parse_args()

    root = Path(args.root)

    if args.check_split:
        check_split(root)
        raise SystemExit(0)

    print(f"=== shock_dataset self-test  (task={args.task}) ===\n")

    # Grid relationship the whole dataset rests on
    lo = load_field(root, 0, 0, 'coarse')
    hi = load_field(root, 0, 0, 'fine')
    err = np.abs(lo - hi[:, ::2, ::2]).max()
    print(f"  t=0 coarse vs fine[::2,::2] : max|diff| = {err:.3e}  "
          f"{'OK (exact node coincidence)' if err == 0 else 'UNEXPECTED'}")

    print("\n  computing train-split statistics...")
    stats = compute_stats(root, DEFAULT_SPLIT['train'], args.task)
    for i, c in enumerate(CHANNELS):
        print(f"    {c:6s} mean={stats['mean'][i]:9.4f}  "
              f"std={stats['std'][i]:8.4f}  res_std={stats['res_std'][i]:8.5f}")

    print()
    ds = ShockDataset(root, DEFAULT_SPLIT['val'], task=args.task,
                      patch=args.patch, random_crop=False)
    x, y = ds[0]
    print(f"\n  item shapes  x={tuple(x.shape)}  y={tuple(y.shape)}")
    print(f"  x  mean={x.mean():+.4f}  std={x.std():.4f}")
    print(f"  y  mean={y.mean():+.4f}  std={y.std():.4f}   (target=residual)")
    print(f"  finite: x={bool(torch.isfinite(x).all())}  "
          f"y={bool(torch.isfinite(y).all())}")

    print(f"  snapshots used: {ds.snapshots[0]}..{ds.snapshots[-1]} "
          f"({len(ds.snapshots)} of {N_SNAPSHOT})")

    # Zero network output must reproduce the input exactly; the resulting
    # error IS the do-nothing baseline the operator has to beat.
    per_frame = []
    for i in range(ds.n_frames):
        xp, yp, _ = ds.full_frame(i)
        per_frame.append(float(np.linalg.norm(yp - xp) / np.linalg.norm(yp)))
    print(f"\n  zero-output baseline relL2 over {ds.n_frames} val frames:")
    print(f"    mean={np.mean(per_frame):.5f}  min={np.min(per_frame):.5f}  "
          f"max={np.max(per_frame):.5f}")
    print("    ^ this is the number the operator has to beat")

    # to_physical must invert the normalisation exactly
    xt   = torch.from_numpy(ds.frames[0][0]).unsqueeze(0)
    yt   = torch.from_numpy(ds.frames[0][1]).unsqueeze(0)
    ytgt = ds[0][1].unsqueeze(0) if args.patch == 0 else None
    if ytgt is not None:
        rec = ds.to_physical(ytgt, xt)
        print(f"\n  to_physical round-trip: max|recon - target| = "
              f"{(rec - yt).abs().max().item():.3e}  (should be ~0)")
