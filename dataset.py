"""
datasets.py — Fast PDE benchmark datasets
==========================================
All solvers are fully vectorised (no Python loops over grid points).
Benchmark timings on a single CPU core at 64×64:
  Darcy    : ~0.05s per sample  (FFT-based spectral solve)
  BentRidge: ~0.01s per sample  (vectorised upwind advection)
  KH       : ~0.05s per sample  (vectorised LLF Euler)

Two usage modes:
  1. On-the-fly (default): data generated at DataLoader construction time.
     Suitable for smoke tests.
  2. Pre-generated (recommended for cluster): call generate_and_save() once,
     then load from .npz files. Use --data_dir in train_all.py.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Darcy Flow  — spectral (FFT) solve, fully vectorised
# ─────────────────────────────────────────────────────────────────────────────

def _solve_darcy_spectral(a: np.ndarray, rng=None,
                           n_iter: int = 30) -> np.ndarray:
    """
    Solve -∇·(a∇u) = f via preconditioned Richardson iteration.

    f is a random zero-mean forcing (required for periodic domain):
    -∇·(a∇u) = f with ∫f = 0 (compatibility condition).

    Using f=constant violates this condition, causing the FFT
    preconditioner to produce u≈0 (all energy sits at DC which
    is zeroed out by the periodic Laplacian).
    """
    H, W = a.shape
    if rng is None:
        rng = np.random.default_rng(0)

    # Zero-mean random forcing (satisfies periodic compatibility condition)
    f = rng.standard_normal((H, W)).astype(np.float64)
    f -= f.mean()

    kx = np.fft.fftfreq(W) * 2 * np.pi
    ky = np.fft.fftfreq(H) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    lap_eig = -(KX**2 + KY**2)
    lap_eig[0, 0] = 1.0

    u = np.zeros((H, W), dtype=np.float64)
    for _ in range(n_iter):
        ux = np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)
        uy = np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)
        flux_x = a * ux; flux_y = a * uy
        div = ((np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) +
               (np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0))) / 4.0
        res = f + div
        res_hat = np.fft.fft2(res); res_hat[0, 0] = 0.0
        u = u + 0.7 * np.fft.ifft2(res_hat / lap_eig).real

    u -= u.mean()
    # Normalise to unit std so all samples have comparable scale
    std = u.std()
    if std > 1e-10:
        u /= std
    return u.astype(np.float32)


def _darcy_coefficient(H, W, rng, n_freq=5):
    a = np.zeros((H, W))
    x = np.linspace(0, 1, W); y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    for _ in range(n_freq):
        kx = rng.integers(1, 4); ky = rng.integers(1, 4)
        phase = rng.uniform(0, 2*np.pi, 2)
        amp   = rng.uniform(0.5, 1.5)
        a += amp * np.cos(2*np.pi*kx*X + phase[0]) * np.cos(2*np.pi*ky*Y + phase[1])
    return np.exp(a).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Bent-Ridge Advection — fully vectorised upwind
# ─────────────────────────────────────────────────────────────────────────────

def _advect_vectorised(u: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                        dt: float, dx: float, n_steps: int) -> np.ndarray:
    """Vectorised upwind advection — no Python loops over grid."""
    for _ in range(n_steps):
        # x-direction upwind (vectorised over full grid)
        du_dx_bwd = (u - np.roll(u, 1,  axis=1)) / dx
        du_dx_fwd = (np.roll(u, -1, axis=1) - u) / dx
        du_dy_bwd = (u - np.roll(u, 1,  axis=0)) / dx
        du_dy_fwd = (np.roll(u, -1, axis=0) - u) / dx

        u = u - dt * (
            np.maximum(cx, 0) * du_dx_bwd + np.minimum(cx, 0) * du_dx_fwd +
            np.maximum(cy, 0) * du_dy_bwd + np.minimum(cy, 0) * du_dy_fwd)
        u = np.clip(u, 0.0, 1.0)
    return u.astype(np.float32)


def _bent_ridge_sample(H, W, rng, n_steps=60):
    x = np.linspace(0, 1, W); y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)

    angle = rng.uniform(0.2, 0.8) * np.pi
    bend  = rng.uniform(0.1, 0.3)
    cx0   = rng.uniform(0.3, 0.5)
    cy0   = rng.uniform(0.3, 0.5)
    front = (np.cos(angle)*(X-cx0) + np.sin(angle)*(Y-cy0) +
             bend * np.sin(2*np.pi*(X+Y)))
    u0 = (front > 0).astype(np.float32)

    omega = rng.uniform(0.5, 1.5)
    cx = -omega * (Y - 0.5)
    cy =  omega * (X - 0.5)

    dx = 1.0 / (W - 1)
    spd = max(np.abs(cx).max(), np.abs(cy).max()) + 1e-12
    dt  = 0.4 * dx / spd

    u = _advect_vectorised(u0.copy(), cx, cy, dt, dx, n_steps)
    return u0, u


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Kelvin-Helmholtz — vectorised LLF finite volume
# ─────────────────────────────────────────────────────────────────────────────

def _kh_sample_fast(H, W, rng, n_steps=120):
    """
    Kelvin-Helmholtz vorticity advection (improved for 128x128).

    Thin shear layers + multi-mode perturbation produce well-developed
    anisotropic vortex rolls. Spectral Poisson for stream function,
    upwind advection for stability.

    Input  = initial vorticity omega_0  (smooth shear layer)
    Target = evolved vorticity omega_T  (developed vortex rolls)
    """
    dx = 1.0 / W
    x  = np.linspace(0, 1, W, endpoint=False)
    y  = np.linspace(0, 1, H, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Thin double shear layer (sigma=0.03 gives sharper, richer instability)
    sigma  = 0.03
    omega0 = (np.exp(-((Y-0.25)**2)/(2*sigma**2)) -
              np.exp(-((Y-0.75)**2)/(2*sigma**2))).astype(np.float64)

    # Multi-mode perturbation to excite several instability wavelengths
    for _ in range(rng.integers(2, 5)):
        k   = rng.integers(2, 6)
        amp = rng.uniform(0.08, 0.20)
        omega0 += amp * np.sin(2*np.pi*k*X + rng.uniform(0, 2*np.pi))

    # Precompute Laplacian eigenvalues (reuse across time steps)
    kx  = np.fft.fftfreq(W) * 2*np.pi
    ky  = np.fft.fftfreq(H) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    lap = -(KX**2 + KY**2); lap[0, 0] = 1.0

    # CFL-stable dt from initial velocity
    psi0 = np.fft.ifft2(np.fft.fft2(-omega0) / lap).real
    spd  = max(
        (np.abs(np.roll(psi0,-1,axis=0) - np.roll(psi0,1,axis=0)) / (2*dx)).max(),
        (np.abs(np.roll(psi0,-1,axis=1) - np.roll(psi0,1,axis=1)) / (2*dx)).max(),
        0.5)
    dt = 0.25 * dx / spd

    omega = omega0.copy()
    for _ in range(n_steps):
        psi  = np.fft.ifft2(np.fft.fft2(-omega) / lap).real
        u    = (np.roll(psi,-1,axis=0) - np.roll(psi,1,axis=0)) / (2*dx)
        v    = -(np.roll(psi,-1,axis=1) - np.roll(psi,1,axis=1)) / (2*dx)
        dxb  = (omega - np.roll(omega,1,axis=1)) / dx
        dxf  = (np.roll(omega,-1,axis=1) - omega) / dx
        dyb  = (omega - np.roll(omega,1,axis=0)) / dx
        dyf  = (np.roll(omega,-1,axis=0) - omega) / dx
        omega = omega - dt * (np.maximum(u,0)*dxb + np.minimum(u,0)*dxf +
                               np.maximum(v,0)*dyb + np.minimum(v,0)*dyf)

    omega0 -= omega0.mean(); omega -= omega.mean()
    return omega0.astype(np.float32), omega.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset classes  (on-the-fly generation)
# ─────────────────────────────────────────────────────────────────────────────

class _NpzDataset(Dataset):
    """Load pre-generated data from a .npz file."""
    def __init__(self, path: Path):
        data = np.load(path)
        self.a = torch.from_numpy(data['a'])   # (N, 1, H, W)
        self.u = torch.from_numpy(data['u'])
    def __len__(self):  return len(self.a)
    def __getitem__(self, i): return self.a[i], self.u[i]


class _GeneratedDataset(Dataset):
    def __init__(self, samples_a, samples_u):
        self.a = [torch.from_numpy(x).unsqueeze(0) for x in samples_a]
        self.u = [torch.from_numpy(x).unsqueeze(0) for x in samples_u]
    def __len__(self):  return len(self.a)
    def __getitem__(self, i): return self.a[i], self.u[i]


def _generate(name, n, H, W, seed):
    rng = np.random.default_rng(seed)
    a_list, u_list = [], []
    print(f"  Generating {n} {name} samples ({H}×{W})...", end='', flush=True)
    skipped = 0
    while len(a_list) < n:
        try:
            if name == 'darcy':
                a = _darcy_coefficient(H, W, rng)
                u = _solve_darcy_spectral(a, rng=rng)
            elif name == 'bentridge':
                a, u = _bent_ridge_sample(H, W, rng)
            elif name == 'kh':
                a, u = _kh_sample_fast(H, W, rng)
            else:
                raise ValueError(f"Unknown dataset: {name}")
            if not np.isfinite(u).all():
                skipped += 1; continue
            a_list.append(a); u_list.append(u)
            if len(a_list) % max(1, n // 5) == 0:
                print(f" {len(a_list)}", end='', flush=True)
        except Exception:
            skipped += 1
    if skipped:
        print(f" ({skipped} skipped)", end='')
    print(" done")
    return a_list, u_list


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-generation: call once, saves .npz files to data_dir
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_save(data_dir: str | Path,
                       datasets=('darcy', 'bentridge', 'kh'),
                       n_train: int = 200, n_test: int = 50,
                       H: int = 64, W: int = 64):
    """
    Pre-generate all datasets and save to .npz files.
    Run this ONCE before submitting training jobs.

    Usage:
        python -c "from datasets import generate_and_save; \\
                   generate_and_save('/scratch/.../data', H=64, W=64)"
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        for split, n, seed in [('train', n_train, 0), ('test', n_test, 9999)]:
            path = data_dir / f'{name}_{split}_{H}x{W}.npz'
            if path.exists():
                print(f"  Skipping {path.name} (already exists)")
                continue
            a_list, u_list = _generate(name, n, H, W, seed)
            a_arr = np.stack(a_list)[:, None]   # (N, 1, H, W)
            u_arr = np.stack(u_list)[:, None]
            np.savez_compressed(path, a=a_arr.astype(np.float32),
                                       u=u_arr.astype(np.float32))
            print(f"  Saved {path}  ({a_arr.shape})")


# ─────────────────────────────────────────────────────────────────────────────
#  Factory  — auto-detects pre-generated files, falls back to on-the-fly
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(dataset_name: str,
                n_train: int = 200, n_test: int = 50,
                H: int = 64, W: int = 64,
                batch_size: int = 8,
                num_workers: int = 0,
                data_dir: str | Path = None) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader).

    If data_dir is given and .npz files exist there, loads from disk
    (fast, ~seconds). Otherwise generates on-the-fly (slow for Darcy).
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        tr_path = data_dir / f'{dataset_name}_train_{H}x{W}.npz'
        te_path = data_dir / f'{dataset_name}_test_{H}x{W}.npz'
        if tr_path.exists() and te_path.exists():
            print(f"  Loading {dataset_name} from {data_dir}")
            train_ds = _NpzDataset(tr_path)
            test_ds  = _NpzDataset(te_path)
            return (DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=num_workers),
                    DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=num_workers))
        else:
            print(f"  .npz not found in {data_dir}, generating on-the-fly")

    a_tr, u_tr = _generate(dataset_name, n_train, H, W, seed=0)
    a_te, u_te = _generate(dataset_name, n_test,  H, W, seed=9999)
    return (DataLoader(_GeneratedDataset(a_tr, u_tr), batch_size=batch_size,
                       shuffle=True,  num_workers=num_workers),
            DataLoader(_GeneratedDataset(a_te, u_te), batch_size=batch_size,
                       shuffle=False, num_workers=num_workers))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--datasets', nargs='+',
                   default=['darcy', 'bentridge', 'kh'])
    p.add_argument('--n_train', type=int, default=200)
    p.add_argument('--n_test',  type=int, default=50)
    p.add_argument('--H',       type=int, default=64)
    p.add_argument('--W',       type=int, default=64)
    args = p.parse_args()
    generate_and_save(args.data_dir, args.datasets,
                      args.n_train, args.n_test, args.H, args.W)