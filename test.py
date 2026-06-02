"""
visualize_predictions.py
========================
Simple single-row visualisation:
  Input | Target | FNO | SNO | USNO | Cascade USNO

Training history visualisation:
  Training and testing relative L2 over epochs for all models

One example from a chosen dataset.

Usage:
    python3 visualize_predictions.py \
        --data_dir  /scratch/project_462001157/SNO/data \
        --ckpt_dir  /scratch/project_462001157/SNO/checkpoints \
        --fig_dir   /scratch/project_462001157/SNO/figures \
        --results_dir /scratch/project_462001157/SNO/results \
        --dataset   darcy \
        --H 64 --W 64 --hidden 32
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────────────────────

BG   = '#0d1117'
AX   = '#161b22'
TEXT = '#e6edf3'
MUT  = '#8b949e'

MCOLS = {'fno': '#58a6ff', 'sno': '#f78166',
         'usno': '#3fb950', 'cascade': '#d2a8ff'}
MLABELS = {'fno': 'FNO', 'sno': 'SNO',
           'usno': 'USNO', 'cascade': 'Cascade USNO'}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2(pred, target):
    return (np.sqrt(np.mean((pred - target)**2)) /
            (np.sqrt(np.mean(target**2)) + 1e-8))


def load_sample(data_dir: Path, dataset: str, H: int, W: int):
    """Load one test sample. Returns (a, u) as (H, W) arrays."""
    for split in ['test', 'train']:
        p = data_dir / f'{dataset}_{split}_{H}x{W}.npz'
        if p.exists():
            d = np.load(p)
            # Pick sample with highest target variance
            u_all = d['u'][:, 0]
            idx   = int(np.argmax(u_all.var(axis=(1, 2))))
            return d['a'][idx, 0], u_all[idx]
    raise FileNotFoundError(f"No .npz for {dataset} in {data_dir}")


def find_checkpoint(ckpt_dir: Path, dataset: str, key: str) -> Path | None:
    stems = [key, key.upper()]
    exts  = ['.pt', '.ckpt']
    for stem in stems:
        for ext in exts:
            for candidate in [ckpt_dir / dataset / f'{stem}{ext}',
                               ckpt_dir / f'{stem}{ext}']:
                if candidate.exists():
                    return candidate
    # glob fallback
    for stem in stems:
        for ext in exts:
            hits = list(ckpt_dir.rglob(f'{stem}{ext}'))
            if hits:
                return hits[0]
    return None


def build_model(key: str, H: int, W: int, hidden: int, n_scales=None):
    from fno          import FNO
    from sno          import SNO
    from usno         import USNO
    from cascade_usno import CascadeUSNO

    n_sh = 1 + int((2**(np.arange(
        int(np.floor(0.5*np.log2(max(H,W)))) if n_scales is None
        else n_scales) + 2)).sum())

    if key == 'fno':
        return FNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                   n_sh=n_sh, n_blocks=4)
    elif key == 'sno':
        return SNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                   n_blocks=4, n_scales=n_scales, input_size=(H, W))
    elif key == 'usno':
        return USNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                    n_scales=n_scales, n_layers=5, input_size=(H, W))
    elif key == 'cascade':
        return CascadeUSNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                           n_scales=n_scales, input_size=(H, W))


def run_inference(key: str, ckpt_dir: Path, dataset: str,
                  a: np.ndarray, H: int, W: int, hidden: int,
                  n_scales=None, device=None):
    """Returns (H, W) prediction array, or None if checkpoint not found."""
    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = find_checkpoint(ckpt_dir, dataset, key)
    if ckpt is None:
        print(f"  [{key}] checkpoint not found")
        return None

    try:
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
        state   = (payload['model_state'] if 'model_state' in payload
                   else payload.get('state_dict', payload))

        # Auto-detect FNO k_max from checkpoint
        local_hidden = hidden
        if key == 'fno':
            w_keys = [k for k in state if 'w_real' in k]
            if w_keys:
                k_max = int(state[w_keys[0]].shape[-1])
                model = (lambda km: __import__('fno', fromlist=['FNO']).FNO(
                    in_channels=1, out_channels=1, hidden_channels=hidden,
                    k_max=km, n_blocks=4))(k_max)
            else:
                model = build_model(key, H, W, hidden, n_scales)
        else:
            model = build_model(key, H, W, hidden, n_scales)

        model.load_state_dict(state)
        model = model.to(device).eval()

        a_t = torch.from_numpy(a[None, None].astype(np.float32)).to(device)
        with torch.no_grad():
            pred = model(a_t).cpu().numpy()[0, 0]

        print(f"  [{key}] OK  range=[{pred.min():.3f}, {pred.max():.3f}]")
        return pred

    except Exception as e:
        import traceback
        print(f"  [{key}] FAILED: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  CSV History Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_history(results_dir: Path, dataset: str, key: str):
    """Load training history CSV for a model. Returns DataFrame or None."""
    csv_path = results_dir / dataset / f'{key}_history.csv'
    if not csv_path.exists():
        print(f"  [{key}] history CSV not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        required_cols = ['epoch', 'train_rl2', 'test_rl2']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  [{key}] CSV missing columns: {missing}")
            return None
        print(f"  [{key}] history loaded: {len(df)} epochs")
        return df
    except Exception as e:
        print(f"  [{key}] FAILED to load history: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_row(a, u, preds, dataset, fig_dir: Path):
    """
    Single row: Input | Target | FNO | SNO | USNO | Cascade USNO
    Below each prediction: absolute error map.
    """
    methods = ['fno', 'sno', 'usno', 'cascade']
    n_cols  = 2 + len(methods)   # input + target + 4 models
    n_rows  = 2                   # field + error

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.4, n_rows * 2.4 + 0.5))
    fig.patch.set_facecolor(BG)
    for ax in axes.ravel():
        ax.set_facecolor(BG); ax.axis('off')

    vlim_u  = max(abs(u.min()), abs(u.max())) or 1.0
    err_max = max([np.abs(preds[m] - u).max()
                   for m in methods if preds.get(m) is not None] + [0.01])

    def show(ax, data, cmap, vmin, vmax, title='', title_col=TEXT):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation='nearest', aspect='equal')
        ax.axis('off')
        if title:
            ax.set_title(title, color=title_col, fontsize=9,
                         pad=3, fontweight='bold')

    # Input
    show(axes[0, 0], a, 'viridis', a.min(), a.max(), title='Input')
    axes[1, 0].set_facecolor(BG)

    # Target
    show(axes[0, 1], u, 'RdBu_r', -vlim_u, vlim_u, title='Target')
    axes[1, 1].set_facecolor(BG)

    # Models
    for col, m in enumerate(methods, start=2):
        pred = preds.get(m)
        col_c = MCOLS[m]
        if pred is not None:
            rl2 = rel_l2(pred, u)
            show(axes[0, col], pred, 'RdBu_r', -vlim_u, vlim_u,
                 title=f'{MLABELS[m]}\nrelL2={rl2:.3f}', title_col=col_c)
            show(axes[1, col], np.abs(pred - u), 'hot', 0, err_max)
        else:
            axes[0, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                              color=MUT, fontsize=11,
                              transform=axes[0, col].transAxes)
            axes[0, col].set_title(MLABELS[m], color=col_c,
                                    fontsize=9, pad=3, fontweight='bold')

    # Row labels
    axes[0, 0].text(-0.08, 0.5, 'Prediction', transform=axes[0, 0].transAxes,
                     ha='right', va='center', rotation=90,
                     color=TEXT, fontsize=8.5)
    axes[1, 0].text(-0.08, 0.5, '|Error|', transform=axes[1, 0].transAxes,
                     ha='right', va='center', rotation=90,
                     color=TEXT, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Dataset: {dataset.upper()}',
                  color=TEXT, fontsize=11, fontweight='bold')

    path = fig_dir / f'{dataset}_single_row.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\nSaved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Training History Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(histories: dict, dataset: str, fig_dir: Path):
    """
    Plot training and testing relative L2 curves over epochs.

    histories: dict mapping model key -> DataFrame with columns
               [epoch, train_rl2, test_rl2, ...]
    """
    methods = ['fno', 'sno', 'usno', 'cascade']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(AX)
        ax.tick_params(colors=TEXT, which='both')
        for spine in ax.spines.values():
            spine.set_color(MUT)
        ax.grid(True, alpha=0.2, color=MUT)

    # Plot each model
    for m in methods:
        df = histories.get(m)
        if df is None or len(df) == 0:
            continue

        col = MCOLS[m]
        label = MLABELS[m]

        # Training curve (solid)
        axes[0].plot(df['epoch'], df['train_rl2'],
                     color=col, linewidth=1.5, label=label, alpha=0.9)
        # Testing curve (dashed)
        axes[1].plot(df['epoch'], df['test_rl2'],
                     color=col, linewidth=1.5, label=label, alpha=0.9,
                     linestyle='--')

    # Configure axes
    axes[0].set_xlabel('Epoch', color=TEXT, fontsize=10)
    axes[0].set_ylabel('Relative L2', color=TEXT, fontsize=10)
    axes[0].set_title('Training Loss', color=TEXT, fontsize=11, fontweight='bold', pad=10)
    axes[0].legend(facecolor=AX, edgecolor=MUT, labelcolor=TEXT,
                   fontsize=9, loc='upper right')
    axes[0].set_yscale('log')

    axes[1].set_xlabel('Epoch', color=TEXT, fontsize=10)
    axes[1].set_ylabel('Relative L2', color=TEXT, fontsize=10)
    axes[1].set_title('Testing Loss', color=TEXT, fontsize=11, fontweight='bold', pad=10)
    axes[1].legend(facecolor=AX, edgecolor=MUT, labelcolor=TEXT,
                   fontsize=9, loc='upper right')
    axes[1].set_yscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(f'Training History — Dataset: {dataset.upper()}',
                  color=TEXT, fontsize=12, fontweight='bold')

    path = fig_dir / f'{dataset}_training_history.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Saved: {path}")
    plt.close()


def plot_combined_history(histories: dict, dataset: str, fig_dir: Path):
    """
    Alternative: train + test on same axes with different line styles.
    """
    methods = ['fno', 'sno', 'usno', 'cascade']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX)
    ax.tick_params(colors=TEXT, which='both')
    for spine in ax.spines.values():
        spine.set_color(MUT)
    ax.grid(True, alpha=0.2, color=MUT)

    for m in methods:
        df = histories.get(m)
        if df is None or len(df) == 0:
            continue

        col = MCOLS[m]
        label = MLABELS[m]

        ax.plot(df['epoch'], df['train_rl2'],
                color=col, linewidth=1.5, alpha=0.9, label=f'{label} (train)')
        ax.plot(df['epoch'], df['test_rl2'],
                color=col, linewidth=1.5, alpha=0.6, linestyle='--',
                label=f'{label} (test)')

    ax.set_xlabel('Epoch', color=TEXT, fontsize=10)
    ax.set_ylabel('Relative L2 (log scale)', color=TEXT, fontsize=10)
    ax.set_title(f'Training & Testing History — {dataset.upper()}',
                  color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.legend(facecolor=AX, edgecolor=MUT, labelcolor=TEXT,
              fontsize=8, loc='upper right', ncol=2)
    ax.set_yscale('log')

    plt.tight_layout()
    path = fig_dir / f'{dataset}_combined_history.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--ckpt_dir',    required=True)
    p.add_argument('--fig_dir',     default='figures')
    p.add_argument('--results_dir', default=None,
                    help='Directory containing results/{dataset}/*.csv files. '
                         'If not set, defaults to parent of data_dir / "results".')
    p.add_argument('--dataset',     default='kh')
    p.add_argument('--H',           type=int, default=128)
    p.add_argument('--W',           type=int, default=128)
    p.add_argument('--hidden',      type=int, default=32)
    p.add_argument('--n_scales',    type=int, default=None)
    p.add_argument('--no_history',  action='store_true',
                    help='Skip training history plots')
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Prediction visualisation ──────────────────────────────────────────
    print(f"\nLoading {args.dataset} sample...")
    a, u = load_sample(Path(args.data_dir), args.dataset, args.H, args.W)
    print(f"  input range=[{a.min():.3f}, {a.max():.3f}]")
    print(f"  target range=[{u.min():.3f}, {u.max():.3f}]")

    print("\nRunning inference...")
    preds = {}
    for key in ['fno', 'sno', 'usno', 'cascade']:
        preds[key] = run_inference(key, Path(args.ckpt_dir), args.dataset,
                                    a, args.H, args.W, args.hidden,
                                    args.n_scales, device)

    print("\nPlotting predictions...")
    plot_row(a, u, preds, args.dataset, fig_dir)

    # ── Training history visualisation ────────────────────────────────────
    if not args.no_history:
        # Determine results directory
        if args.results_dir is not None:
            results_dir = Path(args.results_dir)
        else:
            # Default: sibling of data_dir named "results"
            results_dir = Path(args.data_dir).parent / 'results'

        print(f"\nLoading training histories from {results_dir / args.dataset}...")
        histories = {}
        for key in ['fno', 'sno', 'usno', 'cascade']:
            histories[key] = load_history(results_dir, args.dataset, key)

        # Only plot if at least one history was loaded
        if any(h is not None for h in histories.values()):
            print("\nPlotting training histories...")
            plot_training_history(histories, args.dataset, fig_dir)
            plot_combined_history(histories, args.dataset, fig_dir)
        else:
            print("No training histories found, skipping history plots.")
            print(f"  Expected CSVs at: {results_dir / args.dataset}/*_history.csv")


if __name__ == '__main__':
    main()