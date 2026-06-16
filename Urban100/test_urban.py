"""
test_pl.py — PyTorch Lightning testing for Urban100 inpainting
==============================================================
Loads top-1 checkpoint per method, runs:
  1. Patch-based test via trainer.test()        → test/rl2
  2. Full-image inference (sliding window)      → rl2, PSNR, SSIM
  3. Comparison figures and summary bar charts

Outputs:
    fig_dir/test_summary.png
    fig_dir/test_comparison_NNN.png   (one per test image)
    fig_dir/test_extremes_{key}.png   (best/worst per method)
    out_dir/test_results.csv

Usage:
    python3 test_pl.py \
        --data_dir  /workspace/data/SNO                \
        --ckpt_dir  /workspace/project/SNO/checkpoints \
        --out_dir   /workspace/project/SNO/results     \
        --fig_dir   /workspace/project/SNO/figures     \
        --H 128 --W 128 --hidden 64
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fno_urban           import FNO
from sno_urban           import SNO, build_shearlet_filters
from usno_urban          import USNO
from cascade_usno_urban  import CascadeUSNO
from urban_dataset import Urban100FolderDataset
from torch.utils.data import DataLoader

# Re-import LitModule from train_pl so we can call load_from_checkpoint
from train_urban import InpaintingLitModule, build_model, _gaussian_window


# ─────────────────────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────────────────────

BG   = '#0d1117'; AX = '#161b22'; GRID = '#21262d'
TEXT = '#e6edf3'; MUT = '#8b949e'
COLS = {'fno': '#58a6ff', 'sno': '#f78166',
        'usno': '#3fb950', 'cascade': '#d2a8ff'}
LABS = {'fno': 'FNO', 'sno': 'SNO',
        'usno': 'USNO', 'cascade': 'Cascade USNO'}
ORDER = ['fno', 'sno', 'usno', 'cascade']


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    return (np.sqrt(np.mean((pred - gt) ** 2)) /
            (np.sqrt(np.mean(gt ** 2)) + 1e-8))


def psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 255.0) -> float:
    mse = np.mean((pred - gt) ** 2)
    return 100.0 if mse < 1e-12 else 20.0 * np.log10(max_val / np.sqrt(mse))


def ssim(pred: np.ndarray, gt: np.ndarray, max_val: float = 255.0) -> float:
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * max_val) ** 2, (k2 * max_val) ** 2
    scores = []
    for c in range(pred.shape[0]):
        p, t = pred[c].astype(np.float64), gt[c].astype(np.float64)
        mu_p, mu_t  = p.mean(), t.mean()
        sig_p, sig_t = p.std(), t.std()
        sig_pt = np.mean((p - mu_p) * (t - mu_t))
        scores.append(
            ((2*mu_p*mu_t + c1) * (2*sig_pt + c2)) /
            ((mu_p**2 + mu_t**2 + c1) * (sig_p**2 + sig_t**2 + c2)))
    return float(np.mean(scores))


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {'rl2': rel_l2(pred, gt),
            'psnr': psnr(pred, gt),
            'ssim': ssim(pred, gt)}


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def find_best_ckpt(ckpt_dir: Path, key: str) -> Path | None:
    """
    Priority: lowest val/rl2 in top-k files → last.ckpt
    Lightning names files: {key}-epoch=NNNN-val/rl2=X.XXXXX.ckpt
    The '/' in 'val/rl2' is encoded as 'val' + sep + 'rl2' by Lightning.
    """
    key_dir = ckpt_dir / key
    if not key_dir.exists():
        return None
    # Match any .ckpt not named 'last'
    candidates = [p for p in key_dir.glob('*.ckpt') if p.stem != 'last']
    if candidates:
        # Extract score from filename — last numeric field before .ckpt
        def _score(p):
            try:
                return float(p.stem.split('=')[-1])
            except ValueError:
                return float('inf')
        return min(candidates, key=_score)
    last = key_dir / 'last.ckpt'
    return last if last.exists() else None


def load_lit_module(ckpt_dir: Path, key: str,
                     cfg: dict, H: int, W: int, device) -> InpaintingLitModule | None:
    ckpt = find_best_ckpt(ckpt_dir, key)
    if ckpt is None:
        print(f"  [{key}] no checkpoint found")
        return None
    model = build_model(key, cfg, H, W)
    try:
        lit = InpaintingLitModule.load_from_checkpoint(
            str(ckpt), model=model, cfg=cfg,
            map_location=device)
        lit.eval()
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [{key}] {ckpt.name}  {n:,} params  device={device}")
        return lit
    except Exception as e:
        print(f"  [{key}] FAILED: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Full-image patch inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def patch_inference(model: nn.Module, masked: np.ndarray,
                    H: int, W: int, stride: int, device) -> np.ndarray:
    C, H_img, W_img = masked.shape
    model = model.to(device)          # guarantee model and input are on same device
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


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb(path: Path) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1)


def to_disp(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr.transpose(1, 2, 0), 0, 255).astype(np.uint8)


def err_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt).mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Patch-based test via Lightning trainer
# ─────────────────────────────────────────────────────────────────────────────

def run_patch_test(lit_modules: dict, data_dir: Path,
                    H: int, W: int, batch_size: int) -> dict:
    """Run trainer.test() on urban100_test/ patches for each model."""
    results = {}
    test_ds = Urban100FolderDataset(
        data_dir / 'urban100_test', H=H, W=W, stride=H)
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    for key, lit in lit_modules.items():
        trainer = pl.Trainer(
            accelerator='auto', devices=1,
            logger=False, enable_progress_bar=False,
            enable_model_summary=False)
        out = trainer.test(lit, dataloaders=test_dl, verbose=False)
        results[key] = out[0] if out else {}
        print(f"  [{key}] patch test/rl2 = "
              f"{results[key].get('test/rl2', float('nan')):.5f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Full-image test
# ─────────────────────────────────────────────────────────────────────────────

def run_full_image_test(lit_modules: dict, test_dir: Path,
                         H: int, W: int, device,
                         n_images: int = None) -> list:
    """
    Full-image sliding-window inference on urban100_test/.
    Returns list of per-image result dicts.
    """
    stride     = H // 2
    gt_paths   = sorted((test_dir / 'gt').glob('*.png'))
    msk_paths  = sorted((test_dir / 'masked').glob('*.png'))
    if n_images:
        gt_paths  = gt_paths[:n_images]
        msk_paths = msk_paths[:n_images]

    results = []
    n = len(gt_paths)
    for i, (gp, mp) in enumerate(zip(gt_paths, msk_paths)):
        print(f"  [{i+1}/{n}] {gp.stem}", end='  ', flush=True)
        gt     = load_rgb(gp)
        masked = load_rgb(mp)
        preds  = {}
        mets   = {}

        for key, lit in lit_modules.items():
            pred       = patch_inference(lit.model, masked,
                                          H, W, stride, device)
            preds[key] = pred
            mets[key]  = compute_metrics(pred, gt)

        parts = [f"{LABS.get(k,k)} PSNR={mets[k]['psnr']:.2f}dB"
                 for k in ORDER if k in mets]
        print('  '.join(parts))

        results.append({'name': gp.stem, 'gt': gt, 'masked': masked,
                        'preds': preds, 'metrics': mets})
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(result: dict, fig_dir: Path) -> None:
    gt, masked, preds, mets = (result['gt'], result['masked'],
                                result['preds'], result['metrics'])
    methods = [k for k in ORDER if k in preds]
    n_cols  = 2 + len(methods)

    # Resize for display
    H_img, W_img = gt.shape[1], gt.shape[2]
    scale = min(1.0, 512 / W_img)
    dH, dW = max(1, int(H_img * scale)), max(1, int(W_img * scale))

    def rsz(arr):
        return cv2.resize(to_disp(arr), (dW, dH), interpolation=cv2.INTER_AREA)

    def rsz_e(e):
        return cv2.resize(e, (dW, dH), interpolation=cv2.INTER_AREA)

    err_max = max(err_map(preds[k], gt).max() for k in methods) or 1.0

    fig, axes = plt.subplots(2, n_cols,
        figsize=(n_cols * (dW/100 + 0.2), 2 * (dH/100 + 0.4) + 0.8), dpi=100)
    fig.patch.set_facecolor(BG)
    for ax in axes.ravel():
        ax.set_facecolor(BG); ax.axis('off')

    def show(ax, img, title='', tc=TEXT):
        ax.imshow(rsz(img) if img.ndim == 3 else img,
                  aspect='equal', interpolation='bilinear')
        ax.axis('off')
        if title: ax.set_title(title, color=tc, fontsize=8,
                                pad=3, fontweight='bold')

    def show_e(ax, e, title='', tc=MUT):
        ax.imshow(rsz_e(e), cmap='hot', vmin=0, vmax=err_max,
                  aspect='equal', interpolation='bilinear')
        ax.axis('off')
        if title: ax.set_title(title, color=tc, fontsize=7, pad=2)

    show(axes[0, 0], gt,     'Original')
    show(axes[0, 1], masked, 'Masked input')
    for col, key in enumerate(methods, 2):
        m = mets[key]
        lbl = (f"{LABS[key]}\nRL2={m['rl2']:.4f}"
               f"  PSNR={m['psnr']:.2f}dB\nSSIM={m['ssim']:.4f}")
        show(axes[0, col], preds[key], lbl, COLS[key])

    axes[1, 0].set_facecolor(BG)
    mask_vis = ((masked == 0).all(axis=0).astype(np.uint8) * 255)
    axes[1, 1].imshow(cv2.resize(mask_vis, (dW, dH),
                                  interpolation=cv2.INTER_NEAREST),
                       cmap='gray', vmin=0, vmax=255, aspect='equal')
    axes[1, 1].set_title('Mask', color=MUT, fontsize=7, pad=2)
    for col, key in enumerate(methods, 2):
        show_e(axes[1, col], err_map(preds[key], gt))

    fig.suptitle(f'Urban100 — {result["name"]}',
                  color=TEXT, fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / f'test_comparison_{result["name"]}.png'
    fig.savefig(path, dpi=100, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_summary(results: list, methods: list, fig_dir: Path) -> None:
    mk_names = ['rl2', 'psnr', 'ssim']
    titles   = ['Relative L2 ↓', 'PSNR (dB) ↑', 'SSIM ↑']
    vals = {k: {mk: [r['metrics'][k][mk] for r in results
                      if k in r['metrics']]
                for mk in mk_names}
            for k in methods}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    def ax_s(ax, title=''):
        ax.set_facecolor(AX); ax.tick_params(colors=MUT, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.5, axis='y')
        if title: ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold')

    for col, (mk, title) in enumerate(zip(mk_names, titles)):
        means = [np.mean(vals[k][mk]) for k in methods]
        stds  = [np.std(vals[k][mk])  for k in methods]
        ax = axes[0, col]
        ax.bar(range(len(methods)), means, yerr=stds, capsize=4, width=0.6,
               color=[COLS[k] for k in methods],
               error_kw={'ecolor': MUT, 'lw': 1.2}, alpha=0.88)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABS[k] for k in methods],
                            color=MUT, fontsize=8, rotation=15)
        rng = max(means) - min(means) or 1e-4
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + rng * 0.05,
                    f'{m:.4f}' if mk == 'rl2' else f'{m:.2f}',
                    ha='center', va='bottom', color=TEXT, fontsize=7)
        ax_s(ax, title)

        bp = axes[1, col].boxplot(
            [vals[k][mk] for k in methods], patch_artist=True, widths=0.5,
            medianprops={'color': TEXT, 'lw': 2},
            whiskerprops={'color': MUT}, capprops={'color': MUT},
            flierprops={'marker': 'o', 'markersize': 3,
                        'markerfacecolor': MUT, 'alpha': 0.5})
        for patch, k in zip(bp['boxes'], methods):
            patch.set_facecolor(COLS[k]); patch.set_alpha(0.7)
        axes[1, col].set_xticks(range(1, len(methods)+1))
        axes[1, col].set_xticklabels([LABS[k] for k in methods],
                                      color=MUT, fontsize=8, rotation=15)
        ax_s(axes[1, col], f'{title} distribution')

    fig.suptitle(
        f'Urban100 Inpainting Test  ({len(results)} images)',
        color=TEXT, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = fig_dir / 'test_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  → {path}")


def plot_extremes(results: list, methods: list,
                   fig_dir: Path, n_show: int = 3) -> None:
    for key in methods:
        scored = sorted(
            [(r['metrics'][key]['psnr'], r)
             for r in results if key in r['metrics']],
            key=lambda x: x[0])
        cases = [('Worst', scored[:n_show],  '#f78166'),
                 ('Best',  scored[-n_show:][::-1], '#3fb950')]

        fig, axes = plt.subplots(
            2 * n_show, 4,
            figsize=(14, 4 * n_show),
            gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
        fig.patch.set_facecolor(BG)
        for ax in axes.ravel():
            ax.set_facecolor(BG); ax.axis('off')

        for ci, t in enumerate(['Original','Masked', f'{LABS[key]} Pred','Error']):
            axes[0, ci].set_title(t, color=TEXT, fontsize=9,
                                   fontweight='bold', pad=6)
        row = 0
        for label, case_list, lc in cases:
            for score, r in case_list:
                gt, masked = r['gt'], r['masked']
                pred = r['preds'][key]
                e    = err_map(pred, gt)
                for ci, img in enumerate([gt, masked, pred]):
                    axes[row, ci].imshow(to_disp(img), aspect='equal',
                                          interpolation='bilinear')
                axes[row, 3].imshow(e, cmap='hot', vmin=0, vmax=e.max() or 1,
                                     aspect='equal', interpolation='bilinear')
                m = r['metrics'][key]
                axes[row, 0].set_ylabel(
                    f"{label}\n{r['name']}\nPSNR={m['psnr']:.2f}dB",
                    color=lc, fontsize=7, rotation=0,
                    labelpad=60, va='center')
                row += 1

        fig.suptitle(f'{LABS[key]} — best/worst on Urban100 test',
                      color=TEXT, fontsize=11, fontweight='bold')
        path = fig_dir / f'test_extremes_{key}.png'
        fig.savefig(path, dpi=100, bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f"  → {path.name}")


def save_csv(results: list, methods: list, out_dir: Path) -> None:
    path   = out_dir / 'test_results.csv'
    fields = ['image'] + [f'{k}_{m}' for k in methods
                          for m in ['rl2','psnr','ssim']]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {'image': r['name']}
            for k in methods:
                if k in r['metrics']:
                    for mk, v in r['metrics'][k].items():
                        row[f'{k}_{mk}'] = f'{v:.6f}'
            w.writerow(row)
    print(f"  → {path}")


def print_table(results: list, methods: list) -> None:
    vals = {k: {'rl2': [], 'psnr': [], 'ssim': []} for k in methods}
    for r in results:
        for k in methods:
            if k in r['metrics']:
                for mk in ['rl2', 'psnr', 'ssim']:
                    vals[k][mk].append(r['metrics'][k][mk])
    print(f"\n{'─'*62}")
    print(f"  {'Method':<14} {'Rel-L2 ↓':>10}  {'PSNR ↑':>10}  {'SSIM ↑':>10}")
    print(f"{'─'*62}")
    for k in methods:
        print(f"  {LABS.get(k,k):<14} "
              f"{np.mean(vals[k]['rl2']):>10.5f}  "
              f"{np.mean(vals[k]['psnr']):>10.3f}  "
              f"{np.mean(vals[k]['ssim']):>10.5f}")
    print(f"{'─'*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default='/workspace/data/SNO')
    p.add_argument('--ckpt_dir',  default='/workspace/project/SNO/checkpoints')
    p.add_argument('--out_dir',   default='/workspace/project/SNO/results')
    p.add_argument('--fig_dir',   default='/workspace/project/SNO/figures')
    p.add_argument('--H',         type=int, default=128)
    p.add_argument('--W',         type=int, default=128)
    p.add_argument('--hidden',    type=int, default=32)
    p.add_argument('--n_scales',  type=int, default=None)
    p.add_argument('--n_blocks',  type=int, default=4)
    p.add_argument('--n_layers',  type=int, default=5)
    p.add_argument('--batch',     type=int, default=16)
    p.add_argument('--n_images',  type=int, default=None)
    p.add_argument('--n_extreme', type=int, default=3)
    p.add_argument('--methods', nargs='+', default=None,
                   choices=['fno','sno','usno','cascade'])
    args = p.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir  = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(args.fig_dir);  fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    test_dir = Path(args.data_dir) / 'urban100_test'

    cfg = dict(hidden=args.hidden, n_blocks=args.n_blocks,
               n_layers=args.n_layers, n_scales=args.n_scales,
               lr=1e-3, weight_decay=1e-4, epochs=200)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\nDevice : {device}")
    print(f"Ckpts  : {ckpt_dir}\n")
    print("Loading models...")
    all_lits = {}
    for key in (args.methods or ORDER):
        lit = load_lit_module(ckpt_dir, key, cfg, args.H, args.W, device)
        if lit:
            all_lits[key] = lit
    if not all_lits:
        print("No models loaded."); return
    methods = [k for k in ORDER if k in all_lits]

    # ── Patch-based test (Lightning trainer) ──────────────────────────────────
    print("\nPatch-based test (trainer.test)...")
    run_patch_test(all_lits, Path(args.data_dir), args.H, args.W, args.batch)

    # ── Full-image test ────────────────────────────────────────────────────────
    print(f"\nFull-image test ({test_dir.name}/)...")
    results = run_full_image_test(
        all_lits, test_dir, args.H, args.W, device,
        n_images=args.n_images)

    # ── Metrics ────────────────────────────────────────────────────────────────
    print_table(results, methods)
    save_csv(results, methods, out_dir)

    # ── Figures ────────────────────────────────────────────────────────────────
    print("Saving comparison figures...")
    for r in results:
        plot_comparison(r, fig_dir)

    print("Saving summary...")
    plot_summary(results, methods, fig_dir)

    print("Saving best/worst...")
    plot_extremes(results, methods, fig_dir, n_show=args.n_extreme)

    print(f"\nDone.  results → {out_dir}   figures → {fig_dir}")


if __name__ == '__main__':
    main()