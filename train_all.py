"""
train_all.py — Four-way training comparison: FNO | SNO | USNO | CascadeUSNO
=============================================================================
Usage:
    # Darcy, default settings
    python train_all.py --dataset darcy

    # Full run on all three datasets
    python train_all.py --dataset all --epochs 200 --hidden 32

    # With checkpointing (auto-resume on restart)
    python train_all.py --dataset darcy --ckpt_dir /path/to/ckpts

    # With pre-generated data (recommended on LUMI)
    python train_all.py --dataset darcy --data_dir /path/to/data

Output:
    results/<dataset>/{fno,sno,usno,cascade}_history.csv
    figures/<dataset>_comparison.png
    figures/all_summary.png   (if --dataset all)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fno          import FNO
from sno          import SNO
from usno         import USNO
from cascade_usno import CascadeUSNO, mlmc_band_budget, train_cascade_model
from dataset     import get_loaders


# ─────────────────────────────────────────────────────────────────────────────
#  Plot style constants
# ─────────────────────────────────────────────────────────────────────────────

BG   = '#0d1117'
AX   = '#161b22'
GRID = '#21262d'
TEXT = '#e6edf3'
MUT  = '#8b949e'

COLS = {
    'fno':     '#58a6ff',
    'sno':     '#f78166',
    'usno':    '#3fb950',
    'cascade': '#d2a8ff',
}
LABS = {
    'fno':     'FNO',
    'sno':     'SNO',
    'usno':    'USNO',
    'cascade': 'Cascade USNO',
}


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint save / resume
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(ckpt_dir: Path, key: str, model: nn.Module,
                     opt, sched, hist: dict, extra: dict = None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'opt_state':   opt.state_dict()   if opt   is not None else None,
        'sched_state': sched.state_dict() if sched is not None else None,
        'history':     hist,
        'extra':       extra or {},
    }, ckpt_dir / f'{key}.pt')


def load_checkpoint(ckpt_dir: Path, key: str, model: nn.Module,
                     opt=None, sched=None):
    path = ckpt_dir / f'{key}.pt'
    if not path.exists():
        return None, {}
    payload = torch.load(path, map_location='cpu')
    model.load_state_dict(payload['model_state'])
    if opt   is not None and payload['opt_state']   is not None:
        opt.load_state_dict(payload['opt_state'])
    if sched is not None and payload['sched_state'] is not None:
        sched.load_state_dict(payload['sched_state'])
    ep = payload['history'].get('epoch', [0])
    print(f"  Resumed {key} from epoch {ep[-1] if ep else 0}")
    return payload['history'], payload.get('extra', {})


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt  = target.reshape(target.shape[0], -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + 1e-8)).mean().item()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target)**2).mean() / (target**2).mean().clamp(min=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-end training (FNO / SNO / USNO)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for a, u in loader:
        a, u = a.to(device), u.to(device)
        optimizer.zero_grad()
        pred = model(a)
        loss = mse_loss(pred, u)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += rel_l2(pred.detach(), u)
    return total / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    total = 0.0
    for a, u in loader:
        a, u = a.to(device), u.to(device)
        total += rel_l2(model(a), u)
    return total / len(loader)


def train_standard(model: nn.Module, name: str,
                    train_loader, test_loader,
                    cfg: dict, device,
                    ckpt_dir: Path = None) -> dict:
    opt = optim.AdamW(model.parameters(),
                       lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01)

    key = name.lower()
    hist_loaded, _ = (load_checkpoint(ckpt_dir, key, model, opt, sched)
                      if ckpt_dir else (None, {}))
    if hist_loaded:
        hist     = hist_loaded
        cum      = sum(hist['epoch_time'])
        start_ep = hist['epoch'][-1] + 1
    else:
        hist     = {k: [] for k in ['epoch','train_rl2','test_rl2',
                                      'epoch_time','cumtime']}
        cum      = 0.0
        start_ep = 1

    if start_ep > cfg['epochs']:
        print(f"\n  [{name}] Already complete, skipping.")
        return hist

    print(f"\n  [{name}] {model.param_count():,} params  "
          f"epochs {start_ep} to {cfg['epochs']}")

    for ep in range(start_ep, cfg['epochs'] + 1):
        t0      = time.time()
        tr      = train_epoch(model, train_loader, opt, device)
        te      = evaluate(model, test_loader, device)
        sched.step()
        elapsed = time.time() - t0
        cum    += elapsed

        hist['epoch'].append(ep)
        hist['train_rl2'].append(tr)
        hist['test_rl2'].append(te)
        hist['epoch_time'].append(elapsed)
        hist['cumtime'].append(cum)

        if ep % cfg['log_every'] == 0 or ep == 1:
            print(f"    ep {ep:4d}  train={tr:.5f}  test={te:.5f}  t={elapsed:.1f}s")

        if ckpt_dir and (ep % cfg.get('ckpt_every', 10) == 0 or ep == cfg['epochs']):
            save_checkpoint(ckpt_dir, key, model, opt, sched, hist)

    print(f"  [{name}] Best test relL2 = {min(hist['test_rl2']):.5f}")
    return hist


# ─────────────────────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(hist: dict, path: Path):
    keys = [k for k in hist if isinstance(hist[k], list) and hist[k]]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for i in range(len(hist[keys[0]])):
            w.writerow({k: hist[k][i] for k in keys})


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
    if title:  ax.set_title(title,  color=TEXT, fontsize=9, pad=4, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, color=MUT,  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUT,  fontsize=8)


def plot_comparison(histories: Dict[str, dict],
                     dataset_name: str, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.38, wspace=0.28,
                            left=0.06, right=0.97,
                            top=0.92, bottom=0.05)

    # ── Panel 1: Learning curves ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    for k, hist in histories.items():
        ax.semilogy(hist['epoch'], hist['test_rl2'],
                    color=COLS[k], lw=2.2, label=LABS[k])
        ax.semilogy(hist['epoch'], hist['train_rl2'],
                    color=COLS[k], lw=1, ls='--', alpha=0.45)
    # Shade cascade band phases
    if 'cascade' in histories:
        ch     = histories['cascade']
        bands  = np.array(ch.get('band', []))
        ep_arr = np.array(ch['epoch'])
        K = int(bands[bands >= 0].max()) + 1 if len(bands) and (bands >= 0).any() else 0
        shades = ['#79c0ff','#56d364','#ffa657','#d2a8ff','#ff7b72']
        for k_idx in range(K):
            mask = bands == k_idx
            if mask.any():
                ax.axvspan(ep_arr[mask].min(), ep_arr[mask].max(),
                           alpha=0.08, color=shades[k_idx % len(shades)],
                           label=f'Cascade band {k_idx}')
    ax_style(ax, title=f'Train (--) & Test (─) RelL2  [{dataset_name}]',
             xlabel='Epoch', ylabel='Relative L2')
    ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=AX, edgecolor=GRID,
               loc='upper right', ncol=2)

    # ── Panel 2: Wall-clock ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2:])
    for k, hist in histories.items():
        ax.semilogy(hist['cumtime'], hist['test_rl2'],
                    color=COLS[k], lw=2.2, label=LABS[k])
    ax_style(ax, title='Test RelL2 vs Wall-clock',
             xlabel='Cumulative time (s)', ylabel='Relative L2')
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    # ── Panel 3: Best test error bar ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    keys_ord = [k for k in ['fno','sno','usno','cascade'] if k in histories]
    best     = [min(histories[k]['test_rl2']) for k in keys_ord]
    ax.bar(range(len(keys_ord)), best,
           color=[COLS[k] for k in keys_ord], alpha=0.88, width=0.6)
    ax.set_xticks(range(len(keys_ord)))
    ax.set_xticklabels([LABS[k] for k in keys_ord],
                        color=MUT, fontsize=7.5, rotation=15)
    fno_best = min(histories['fno']['test_rl2']) if 'fno' in histories else best[0]
    for i, (k, v) in enumerate(zip(keys_ord, best)):
        imp = (fno_best - v) / (fno_best + 1e-8) * 100
        lbl = f'{v:.4f}' if k == 'fno' else f'{v:.4f}\n({imp:+.0f}%)'
        ax.text(i, v * 1.02, lbl, ha='center', va='bottom',
                color=TEXT, fontsize=7)
    ax_style(ax, title='Best Test RelL2', ylabel='Relative L2')
    ax.set_ylim(0, max(best) * 1.4)

    # ── Panel 4: Convergence rate ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for k, hist in histories.items():
        ep = np.array(hist['epoch']); te = np.array(hist['test_rl2'])
        w  = max(5, len(ep) // 10)
        slopes = [(np.log(te[i]) - np.log(te[i-w])) / max(ep[i]-ep[i-w], 1)
                  for i in range(w, len(ep))]
        if slopes:
            ax.plot(ep[w:], slopes, color=COLS[k], lw=1.8, label=LABS[k])
    ax.axhline(0, color=GRID, lw=0.8, ls='--')
    ax_style(ax, title='Convergence Rate\n(d log relL2 / d epoch)',
             xlabel='Epoch', ylabel='Rate')
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    # ── Panel 5: Per-epoch time ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for k, hist in histories.items():
        ep = np.array(hist['epoch']); et = np.array(hist['epoch_time'])
        w  = max(3, len(ep) // 15)
        sm = np.convolve(et, np.ones(w)/w, mode='valid')
        ax.plot(ep[:len(sm)], sm, color=COLS[k], lw=1.8, label=LABS[k])
    ax_style(ax, title='Per-epoch Wall Time',
             xlabel='Epoch', ylabel='Time (s)')
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    # ── Panel 6: Generalisation gap ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    for k, hist in histories.items():
        ep  = np.array(hist['epoch'])
        gap = np.array(hist['test_rl2']) - np.array(hist['train_rl2'])
        ax.plot(ep, gap, color=COLS[k], lw=1.8, label=LABS[k])
    ax.axhline(0, color=GRID, lw=0.8, ls='--')
    ax_style(ax, title='Generalisation Gap\n(test − train)',
             xlabel='Epoch', ylabel='Gap')
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    # ── Panel 7: Cascade band residuals ──────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    if 'cascade' in histories:
        ch    = histories['cascade']
        ep    = np.array(ch['epoch'])
        bands = np.array(ch.get('band', [-1]*len(ep)))
        bsig  = np.array(ch.get('band_sigma', ch['train_rl2']))
        K_max = int(bands[bands >= 0].max()) + 1 if (bands >= 0).any() else 1
        bcols = ['#79c0ff','#56d364','#ffa657','#d2a8ff','#ff7b72']
        for k_idx in range(K_max):
            mask = bands == k_idx
            if mask.any():
                ax.semilogy(ep[mask], bsig[mask],
                            color=bcols[k_idx % len(bcols)],
                            lw=2, label=f'Band {k_idx}')
        vsweep = bands < 0
        if vsweep.any():
            ax.semilogy(ep[vsweep], bsig[vsweep],
                        color='#ffa657', lw=1.5, ls=':', label='V-sweep')
        ax_style(ax, title='Cascade: Per-band Residual (greedy decay)',
                 xlabel='Epoch', ylabel='Band residual relL2')
        ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)
    else:
        ax.set_facecolor(BG); ax.axis('off')
        ax.text(0.5, 0.5, 'CascadeUSNO not trained',
                ha='center', va='center', color=MUT, fontsize=10,
                transform=ax.transAxes)

    # ── Panel 8: MLMC budget ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    if 'cascade' in histories:
        ch     = histories['cascade']
        bands  = np.array(ch.get('band', []))
        K_max  = int(bands[bands >= 0].max()) + 1 if len(bands) and (bands >= 0).any() else 1
        actual = [int((bands == k).sum()) for k in range(K_max)]
        raw    = [2**(-(2+2)*k/2) for k in range(1, K_max+1)]
        ideal  = [r/sum(raw)*sum(actual) for r in raw]
        x      = np.arange(K_max)
        ax.bar(x - 0.18, actual, 0.35, color='#56d364', alpha=0.85, label='Actual')
        ax.bar(x + 0.18, ideal,  0.35, color='#d2a8ff', alpha=0.85, label='MLMC theory')
        ax.set_xticks(x)
        ax.set_xticklabels([f'band {k}' for k in range(K_max)],
                            color=MUT, fontsize=8)
    ax_style(ax, title='Cascade Budget vs MLMC Theory',
             xlabel='Band', ylabel='Epochs')
    ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    # ── Panel 9: Summary table ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    rows = []
    for k in [k for k in ['fno','sno','usno','cascade'] if k in histories]:
        h   = histories[k]
        ep  = np.array(h['epoch']); te = np.array(h['test_rl2'])
        rows.append([LABS[k], f"{min(te):.4f}", f"{ep[np.argmin(te)]}",
                     f"{sum(h['epoch_time']):.0f}s"])
    if rows:
        tbl = ax.table(cellText=rows,
                        colLabels=['Model','Best RelL2','Best Ep','Total Time'],
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#21262d' if r == 0 else AX)
            cell.set_text_props(color=TEXT)
            cell.set_edgecolor(GRID)
    ax.set_title('Summary', color=TEXT, fontsize=9, pad=3, fontweight='bold')

    fig.suptitle(f'FNO | SNO | USNO | Cascade USNO  —  {dataset_name.upper()}',
                  color=TEXT, fontsize=12, fontweight='bold', y=0.975)
    path = fig_dir / f'{dataset_name}_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {path}")
    plt.close()


def plot_all_summary(all_histories: Dict[str, Dict[str, dict]], fig_dir: Path):
    datasets = list(all_histories.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5))
    fig.patch.set_facecolor(BG)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for k, hist in all_histories[ds].items():
            ax.semilogy(hist['epoch'], hist['test_rl2'],
                        color=COLS[k], lw=2.2, label=LABS[k])
        ax_style(ax, title=ds.upper(), xlabel='Epoch', ylabel='Test RelL2')
        ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)
    fig.suptitle('Cross-dataset Summary', color=TEXT,
                  fontsize=12, fontweight='bold')
    path = fig_dir / 'all_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: number of shearlets for a given grid
# ─────────────────────────────────────────────────────────────────────────────

def _get_n_sh(H: int, W: int, n_scales: int = None) -> int:
    if n_scales is None:
        n_scales = int(np.floor(0.5 * np.log2(max(H, W))))
    return 1 + int((2**(np.arange(n_scales) + 2)).sum())


# ─────────────────────────────────────────────────────────────────────────────
#  Main run function
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(cfg: dict) -> Dict[str, dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds     = cfg['dataset']
    H, W   = cfg['H'], cfg['W']
    sz     = (H, W)

    print(f"\n{'='*60}")
    print(f"Dataset: {ds.upper()}   Grid: {H}x{W}   Device: {device}")
    print(f"Epochs: {cfg['epochs']}   Hidden: {cfg['hidden']}   Batch: {cfg['batch_size']}")
    print(f"{'='*60}")

    train_loader, test_loader = get_loaders(
        ds, n_train=cfg['n_train'], n_test=cfg['n_test'],
        H=H, W=W, batch_size=cfg['batch_size'],
        data_dir=cfg.get('data_dir'))

    n_sh    = _get_n_sh(H, W, cfg.get('n_scales'))
    methods = cfg.get('methods', ['fno','sno','usno','cascade'])
    _all_models = {
        'fno': FNO(
            in_channels=1, out_channels=1,
            hidden_channels=cfg['hidden'],
            k_max=cfg.get('k_max'),
            n_sh=n_sh,
            n_blocks=cfg['n_blocks']).to(device),
        'sno': SNO(
            in_channels=1, out_channels=1,
            hidden_channels=cfg['hidden'],
            n_scales=cfg.get('n_scales'),
            n_blocks=cfg['n_blocks'],
            input_size=sz).to(device),
        'usno': USNO(
            in_channels=1, out_channels=1,
            hidden_channels=cfg['hidden'],
            n_scales=cfg.get('n_scales'),
            n_layers=cfg['n_layers'],
            input_size=sz).to(device),
        'cascade': CascadeUSNO(
            in_channels=1, out_channels=1,
            hidden_channels=cfg['hidden'],
            n_scales=cfg.get('n_scales'),
            input_size=sz).to(device),
    }

    models = {k: v for k, v in _all_models.items() if k in methods}
    print(f"\nParameter counts:")
    for k, m in models.items():
        print(f"  {k:10s}: {m.param_count():>8,}")

    out_dir  = Path(cfg['out_dir']) / ds
    fig_dir  = Path(cfg['fig_dir'])
    ckpt_dir = Path(cfg['ckpt_dir']) / ds if cfg.get('ckpt_dir') else None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    histories = {}

    # Train FNO, SNO, USNO end-to-end
    methods = cfg.get('methods', ['fno','sno','usno','cascade'])
    for key in ['fno', 'sno', 'usno']:
        if key not in methods:
            continue
        hist = train_standard(models[key], key.upper(),
                               train_loader, test_loader,
                               cfg, device, ckpt_dir=ckpt_dir)
        histories[key] = hist
        save_csv(hist, out_dir / f'{key}_history.csv')
        try:
            plot_comparison(histories, ds, fig_dir)
        except Exception as e:
            print(f"  [plot] {e}")

    # Train CascadeUSNO with greedy band training
    if 'cascade' in methods:
      hist_c = train_cascade_model(models['cascade'], train_loader, test_loader,
                                    cfg, device, ckpt_dir=ckpt_dir)
      histories['cascade'] = hist_c
      save_csv(hist_c, out_dir / 'cascade_history.csv')
      try:
          plot_comparison(histories, ds, fig_dir)
      except Exception as e:
          print(f"  [plot] {e}")

    # Summary
    print(f"\n{'─'*50}")
    print(f"{'Model':<14} {'Best RelL2':>12} {'Best Ep':>8}")
    print('─' * 36)
    fno_best = min(histories['fno']['test_rl2']) if 'fno' in histories else None
    for key in ['fno','sno','usno','cascade']:
        if key not in histories:
            continue
        h    = histories[key]
        best = min(h['test_rl2'])
        bep  = h['epoch'][np.argmin(h['test_rl2'])]
        tag  = (f' ({(fno_best-best)/(fno_best+1e-8)*100:+.1f}%)'
                if fno_best and key != 'fno' else '')
        print(f"  {key:<12} {best:>12.5f}{tag}  ep {bep}")

    return histories


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    'dataset'      : 'bentridge',
    'H'            : 64,
    'W'            : 64,
    'n_train'      : 200,
    'n_test'       : 50,
    'batch_size'   : 8,
    'epochs'       : 200,
    'lr'           : 1e-3,
    'weight_decay' : 1e-4,
    'hidden'       : 32,
    'k_max'        : None,     # auto-matched to SNO param count
    'n_scales'     : None,     # auto: floor(0.5 * log2(max(H,W)))
    'n_blocks'     : 4,        # FNO + SNO
    'n_layers'     : 5,        # USNO
    'mlmc_alpha'   : 2.0,      # CascadeUSNO MLMC exponent
    'sobolev_s'    : 0.0,      # CascadeUSNO Sobolev weight exponent
    'log_every'    : 10,
    'out_dir'      : 'results',
    'fig_dir'      : 'figures',
    'ckpt_dir'     : None,
    'ckpt_every'   : 10,
    'data_dir'     : None,
    'methods'      : ['fno', 'sno', 'usno', 'cascade'],
}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='FNO | SNO | USNO | CascadeUSNO comparison')
    p.add_argument('--dataset',    default='bentridge',
                   choices=['darcy','bentridge','kh','all'])
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--H',          type=int,   default=128)
    p.add_argument('--W',          type=int,   default=128)
    p.add_argument('--hidden',     type=int,   default=32)
    p.add_argument('--n_blocks',   type=int,   default=4)
    p.add_argument('--n_layers',   type=int,   default=5)
    p.add_argument('--k_max',      type=int,   default=None)
    p.add_argument('--n_scales',   type=int,   default=None)
    p.add_argument('--batch',      type=int,   default=None)
    p.add_argument('--lr',         type=float, default=None)
    p.add_argument('--out_dir',    default='results')
    p.add_argument('--fig_dir',    default='figures')
    p.add_argument('--ckpt_dir',   default='checkpoints')
    p.add_argument('--ckpt_every', type=int, default=10)
    p.add_argument('--data_dir',   default='data')
    p.add_argument('--methods', nargs='+',
                   default=None,
                   choices=['fno','sno','usno','cascade'],
                   help='Which methods to train. Default: all four.')
    args = p.parse_args()

    cfg = DEFAULT_CFG.copy()
    cfg['out_dir']    = args.out_dir
    cfg['fig_dir']    = args.fig_dir
    cfg['ckpt_dir']   = args.ckpt_dir
    cfg['ckpt_every'] = args.ckpt_every
    cfg['data_dir']   = args.data_dir
    if args.methods:
        cfg['methods'] = args.methods
    if args.epochs:   cfg['epochs']    = args.epochs
    if args.H:        cfg['H'] = cfg['W'] = args.H
    if args.W:        cfg['W']         = args.W
    if args.hidden:   cfg['hidden']    = args.hidden
    if args.k_max:    cfg['k_max']     = args.k_max
    if args.n_scales: cfg['n_scales']  = args.n_scales
    if args.n_blocks: cfg['n_blocks']  = args.n_blocks
    if args.n_layers: cfg['n_layers']  = args.n_layers
    if args.batch:    cfg['batch_size'] = args.batch
    if args.lr:       cfg['lr']        = args.lr

    Path(cfg['out_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['fig_dir']).mkdir(parents=True, exist_ok=True)

    datasets = (['darcy','bentridge','kh']
                if args.dataset == 'all' else [args.dataset])

    all_histories = {}
    for ds in datasets:
        cfg['dataset'] = ds
        all_histories[ds] = run_dataset(cfg)

    if len(datasets) > 1:
        plot_all_summary(all_histories, Path(cfg['fig_dir']))

    print('\n\nFINAL SUMMARY')
    print('=' * 62)
    print(f"{'Dataset':<12} {'FNO':>10} {'SNO':>10} {'USNO':>10} {'Cascade':>10}")
    print('-' * 55)
    for ds, hists in all_histories.items():
        row = f"{ds:<12}"
        for k in ['fno','sno','usno','cascade']:
            row += f" {min(hists[k]['test_rl2']):>10.5f}" if k in hists else f"{'N/A':>10}"
        print(row)