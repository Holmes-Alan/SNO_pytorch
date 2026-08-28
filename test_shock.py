"""
test_shock.py — evaluate trained operators on the shock-bubble test split
==========================================================================
Loads the best checkpoint per method, runs FULL-FRAME inference on the held-
out samples, and reports fine-detail-sensitive metrics plus figures.

Deliberately free of any PyTorch Lightning dependency: the checkpoint is read
with plain torch.load, so this runs anywhere torch does and cannot break on a
Lightning version bump.

METRICS
-------
rl2           global relative L2.  Dominated by the large smooth regions —
              every method scores well here, so it HIDES the effect this
              study is about.  Reported for completeness, not as the headline.
rl2_highpass  relative L2 above 0.25 x Nyquist.  ~7x more discriminating than
              global rl2 on this data (measured on the do-nothing baseline).
rl2_front     relative L2 on the top 10% of |grad rho| — i.e. on the shock
              fronts themselves.  Mask comes from the TARGET, so it is
              identical for every method.
mass_err      relative error in total mass; a conservation sanity check.
skill         1 - rl2/baseline_rl2.  Fraction of the coarse solver's error
              the operator actually removed.  0 = did nothing, 1 = perfect.

The `bicubic`/`coarse` do-nothing baseline is always included as a column, so
a model that fails to beat it is immediately visible.

USAGE
-----
    python3 test_shock.py --task same \
        --ckpt_dir checkpoints/shock --out_dir results/shock \
        --fig_dir figures/shock
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shock_common as SC
from shock_common import (build_model, all_metrics, predict_frame,
                          radial_spectrum, rel_l2)
from shock_dataset import (ShockDataset, DEFAULT_SPLIT, CHANNELS,
                           default_root, grid_size, load_or_compute_stats)

BG, AX, GRID = '#0d1117', '#161b22', '#21262d'
TEXT, MUT    = '#e6edf3', '#8b949e'
BASELINE_KEY = 'baseline'
BASELINE_LAB = 'do nothing'


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def find_best_ckpt(ckpt_dir: Path, name: str) -> Optional[Path]:
    """
    Lowest val_rl2 among the top-k files, else last.ckpt.

    Uses rglob, not glob: older runs whose filename template contained a '/'
    (from monitoring 'val/rl2') had Lightning silently create a DIRECTORY per
    checkpoint, hiding the .ckpt one level down.  train_shock.py logs an
    underscore key to avoid that, but we stay tolerant of the old layout.
    """
    d = Path(ckpt_dir) / name
    if not d.exists():
        return None
    cands = [p for p in d.rglob('*.ckpt') if p.stem != 'last']
    if cands:
        def score(p: Path) -> float:
            try:
                return float(p.stem.split('-')[-1].split('=')[-1])
            except ValueError:
                return float('inf')
        best = min(cands, key=score)
        if score(best) < float('inf'):
            return best
    last = d / 'last.ckpt'
    return last if last.exists() else None


def load_operator(ckpt_path: Path, cfg: dict, n_in: int, device
                  ) -> Tuple[torch.nn.Module, dict, Optional[dict]]:
    """
    Rebuild the operator and load weights from a Lightning checkpoint.

    Returns (model, cfg_used, stats_from_ckpt).  Config and normalisation
    statistics stored inside the checkpoint take precedence over the CLI, so
    evaluation cannot silently use a different architecture or normalisation
    than the one that was trained.
    """
    payload = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    hp      = payload.get('hyper_parameters', {}) or {}
    ck_cfg  = dict(hp.get('cfg', {}) or {})
    stats   = hp.get('stats')

    merged = dict(cfg)
    for k in ('hidden', 'n_blocks', 'n_layers', 'n_scales', 'k_max', 'pad',
              'patch', 'task', 'target', 'in_channels', 'out_channels',
              'lifting_dim', 'n_levels', 'num_heads', 'groups',
              'warp_guided', 'max_disp', 'warp_energy'):
        if k in ck_cfg and ck_cfg[k] is not None:
            merged[k] = ck_cfg[k]

    key   = cfg['_key']
    # Rebuild at the geometry the model was TRAINED at.  The shearlet bank is
    # baked in at construction, so a 128-patch operator cannot be instantiated
    # at 257²; the checkpoint's patch therefore wins over the CLI's.
    n_use = merged.get('patch') or n_in
    merged['_n_in'] = n_use
    model = build_model(key, merged, n_use)

    sd  = payload.get('state_dict', payload)
    own = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(own, strict=False)
    if missing or unexpected:
        print(f"    warning: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model.to(device).eval(), merged, stats


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(models: Dict[str, torch.nn.Module],
             cfgs: Dict[str, dict],
             ds: ShockDataset,
             device,
             max_frames: Optional[int] = None) -> List[dict]:
    """One record per test frame, holding metrics and predictions per method."""
    n    = ds.n_frames if max_frames is None else min(max_frames, ds.n_frames)
    out  = []
    for i in range(n):
        x_phys, y_phys, (s, k) = ds.full_frame(i)
        xt = torch.from_numpy(x_phys).unsqueeze(0)
        yt = torch.from_numpy(y_phys).unsqueeze(0)

        preds = {BASELINE_KEY: x_phys}                 # do nothing
        mets  = {BASELINE_KEY: all_metrics(xt, yt)}

        for key, model in models.items():
            c = cfgs[key]
            p = predict_frame(model, x_phys, ds.stats, target=c['target'],
                              patch=c['patch'],
                              stride=(c['patch'] // 2) if c['patch'] else None,
                              device=device)
            preds[key] = p
            mets[key]  = all_metrics(torch.from_numpy(p).unsqueeze(0), yt)

        base = mets[BASELINE_KEY]['rl2']
        for key in mets:
            mets[key]['skill'] = 1.0 - mets[key]['rl2'] / max(base, 1e-12)

        out.append({'sample': s, 'snapshot': k, 'gt': y_phys,
                    'preds': preds, 'metrics': mets})
        print(f"  [{i+1}/{n}] s{s:02d} t{k:02d}  " +
              "  ".join(f"{SC.LABELS.get(m, m)}={mets[m]['rl2']:.4f}"
                        for m in mets))
    return out


def aggregate(results: List[dict], methods: List[str]) -> Dict[str, Dict[str, float]]:
    keys = list(results[0]['metrics'][methods[0]].keys())
    return {m: {k: float(np.nanmean([r['metrics'][m][k] for r in results]))
                for k in keys}
            for m in methods}


def print_table(agg: Dict[str, Dict[str, float]], methods: List[str],
                params: Optional[Dict[str, int]] = None) -> None:
    cols = ['rl2', 'rl2_highpass', 'rl2_front', 'skill', 'mass_err']
    params = params or {}
    w = 100 if params else 86
    print('\n' + '─' * w)
    print(f"  {'Method':<16}" + (f"{'params':>11}" if params else '')
          + "".join(f"{c:>14}" for c in cols))
    print('─' * w)
    for m in methods:
        lab = BASELINE_LAB if m == BASELINE_KEY else SC.LABELS.get(m, m)
        pc  = f"{params[m]:>11,}" if m in params else (' ' * 11 if params else '')
        print(f"  {lab:<16}{pc}" + "".join(f"{agg[m][c]:>14.5f}" for c in cols))
    print('─' * w)
    print("  rl2_highpass and rl2_front are the fine-detail metrics; global")
    print("  rl2 is dominated by the smooth bulk and understates differences.\n")


def save_csv(results: List[dict], methods: List[str], path: Path) -> None:
    mkeys  = list(results[0]['metrics'][methods[0]].keys())
    fields = ['sample', 'snapshot'] + [f'{m}_{k}' for m in methods for k in mkeys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {'sample': r['sample'], 'snapshot': r['snapshot']}
            for m in methods:
                for k, v in r['metrics'][m].items():
                    row[f'{m}_{k}'] = f'{v:.6f}'
            w.writerow(row)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX)
    ax.tick_params(colors=MUT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
    if title:  ax.set_title(title,  color=TEXT, fontsize=9, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, color=MUT, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=MUT, fontsize=8)


# Distinct colours for derived runs (fno_wide, sno_deep, ...) that are not
# one of the four canonical method names.  Deliberately excludes near-white,
# which is reserved for the ground-truth curve in the spectrum plot.
_EXTRA_COLORS = ('#ffa657', '#79c0ff', '#56d364', '#ff7b72', '#a5d6ff')


def _color(m: str) -> str:
    if m == BASELINE_KEY:
        return MUT
    if m in SC.COLORS:
        return SC.COLORS[m]
    # Deterministic across runs — unlike hash(), which is salted per process.
    return _EXTRA_COLORS[sum(map(ord, m)) % len(_EXTRA_COLORS)]


def _label(m: str) -> str:
    return BASELINE_LAB if m == BASELINE_KEY else SC.LABELS.get(m, m)


def plot_summary(results: List[dict], methods: List[str], fig_dir: Path) -> None:
    cols   = ['rl2', 'rl2_highpass', 'rl2_front']
    titles = ['Global relative L2 ↓\n(dominated by smooth bulk)',
              'High-pass relative L2 ↓\n(>0.25 Nyquist — fine detail)',
              'Shock-front relative L2 ↓\n(top 10% of |∇ρ|)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(BG)

    for c, (metric, title) in enumerate(zip(cols, titles)):
        vals  = {m: [r['metrics'][m][metric] for r in results] for m in methods}
        means = [np.mean(vals[m]) for m in methods]
        stds  = [np.std(vals[m])  for m in methods]

        ax = axes[0, c]
        ax.bar(range(len(methods)), means, yerr=stds, capsize=4, width=0.62,
               color=[_color(m) for m in methods], alpha=0.9,
               error_kw={'ecolor': MUT, 'lw': 1.1})
        if BASELINE_KEY in methods:
            ax.axhline(means[methods.index(BASELINE_KEY)],
                       color=MUT, ls='--', lw=1.0)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m) for m in methods],
                           color=MUT, fontsize=8, rotation=18)
        span = (max(means) - min(means)) or 1e-4
        for i, (mu, sd) in enumerate(zip(means, stds)):
            ax.text(i, mu + sd + span * 0.06, f'{mu:.4f}',
                    ha='center', va='bottom', color=TEXT, fontsize=7)
        _style(ax, title)

        bp = axes[1, c].boxplot([vals[m] for m in methods], patch_artist=True,
                                widths=0.5,
                                medianprops={'color': TEXT, 'lw': 2},
                                whiskerprops={'color': MUT},
                                capprops={'color': MUT},
                                flierprops={'marker': 'o', 'markersize': 3,
                                            'markerfacecolor': MUT, 'alpha': 0.5})
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(_color(m)); patch.set_alpha(0.7)
        axes[1, c].set_xticks(range(1, len(methods) + 1))
        axes[1, c].set_xticklabels([_label(m) for m in methods],
                                   color=MUT, fontsize=8, rotation=18)
        _style(axes[1, c], 'distribution over test frames')

    fig.suptitle(f'Shock-bubble coarse→fine  ({len(results)} test frames)',
                 color=TEXT, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = fig_dir / 'shock_summary.png'
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def plot_spectra(results: List[dict], methods: List[str], fig_dir: Path,
                 channel: int = 0) -> None:
    """
    Radially averaged power spectrum, averaged over test frames.

    This is the fine-detail figure: a model that smooths out shock structure
    has a spectrum that falls below the truth at high wavenumber, and the gap
    is visible directly rather than hidden inside a scalar.
    """
    spec_gt, spec_m = [], {m: [] for m in methods}
    for r in results:
        k, s = radial_spectrum(r['gt'][channel])
        spec_gt.append(s)
        for m in methods:
            spec_m[m].append(radial_spectrum(r['preds'][m][channel])[1])
    k       = radial_spectrum(results[0]['gt'][channel])[0]
    gt_mean = np.mean(spec_gt, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.loglog(k[1:], gt_mean[1:], color=TEXT, lw=2.4, label='fine (truth)')
    for m in methods:
        ax.loglog(k[1:], np.mean(spec_m[m], axis=0)[1:],
                  color=_color(m), lw=1.6, label=_label(m), alpha=0.9)
    _style(ax, f'Radial power spectrum — {CHANNELS[channel]}',
           'wavenumber (cycles/px)', 'power')
    ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    ax = axes[1]
    for m in methods:
        ratio = np.mean(spec_m[m], axis=0)[1:] / np.maximum(gt_mean[1:], 1e-30)
        ax.semilogx(k[1:], ratio, color=_color(m), lw=1.8,
                    label=_label(m), alpha=0.9)
    ax.axhline(1.0, color=TEXT, lw=1.2, ls='--')
    ax.set_ylim(0, 2)
    _style(ax, 'Spectral ratio  pred / truth\n(<1 = detail smoothed away)',
           'wavenumber (cycles/px)', 'ratio')
    ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    fig.suptitle('Where each operator puts its energy',
                 color=TEXT, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = fig_dir / 'shock_spectrum.png'
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def plot_frame(result: dict, methods: List[str], fig_dir: Path,
               channel: int = 0) -> None:
    gt   = result['gt'][channel]
    ncol = 1 + len(methods)
    errs = {m: np.abs(result['preds'][m][channel] - gt) for m in methods}
    emax = max(e.max() for e in errs.values()) or 1.0

    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.6))
    fig.patch.set_facecolor(BG)
    for ax in np.ravel(axes):
        ax.set_facecolor(BG); ax.axis('off')

    axes[0, 0].imshow(gt, origin='lower', cmap='viridis')
    axes[0, 0].set_title('fine (truth)', color=TEXT, fontsize=9,
                         fontweight='bold')
    axes[1, 0].axis('off')

    for c, m in enumerate(methods, 1):
        met = result['metrics'][m]
        axes[0, c].imshow(result['preds'][m][channel], origin='lower',
                          cmap='viridis')
        axes[0, c].set_title(
            f"{_label(m)}\nrL2={met['rl2']:.4f}  hp={met['rl2_highpass']:.4f}",
            color=_color(m), fontsize=8, fontweight='bold')
        axes[1, c].imshow(errs[m], origin='lower', cmap='hot',
                          vmin=0, vmax=emax)
        axes[1, c].set_title('|error|', color=MUT, fontsize=7)

    fig.suptitle(f"sample {result['sample']:02d}  snapshot {result['snapshot']:02d}"
                 f"  —  {CHANNELS[channel]}",
                 color=TEXT, fontsize=11, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    path = fig_dir / f"shock_frame_s{result['sample']:02d}_t{result['snapshot']:02d}.png"
    fig.savefig(path, dpi=110, facecolor=BG, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',     default=default_root())
    p.add_argument('--ckpt_dir', default='checkpoints/shock')
    p.add_argument('--out_dir',  default='results/shock')
    p.add_argument('--fig_dir',  default='figures/shock')
    p.add_argument('--task',     default='same', choices=['same', 'sr'])
    p.add_argument('--target',   default='residual', choices=['residual', 'direct'])
    p.add_argument('--split_json', default=None)
    p.add_argument('--patch',    type=int, default=0)
    p.add_argument('--pad',      type=int, default=16)
    p.add_argument('--hidden',   type=int, default=32)
    p.add_argument('--n_blocks', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--n_scales', type=int, default=None)
    p.add_argument('--k_max',    type=int, default=None)
    p.add_argument('--max_frames', type=int, default=None)
    p.add_argument('--n_frame_figs', type=int, default=6)
    p.add_argument('--names', nargs='+', default=None,
                   help='checkpoint dir names to evaluate; defaults to the '
                        'four methods. Use e.g. fno sno fno_wide to include '
                        'the bandwidth control.')
    args = p.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root     = Path(args.root)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)

    split = dict(DEFAULT_SPLIT)
    if args.split_json:
        with open(args.split_json) as f:
            split = json.load(f)

    n_grid = grid_size(args.task)
    n_in   = args.patch or n_grid
    base_cfg = dict(task=args.task, target=args.target, patch=args.patch,
                    pad=args.pad, hidden=args.hidden, n_blocks=args.n_blocks,
                    n_layers=args.n_layers, n_scales=args.n_scales,
                    k_max=args.k_max, in_channels=4, out_channels=4,
                    lifting_dim=None, n_levels=None, num_heads=None,
                    groups=None)

    print(f"\ndevice : {device}")
    print(f"ckpts  : {ckpt_dir}")
    print(f"task   : {args.task}   grid {n_grid}²   model input {n_in}²\n")

    names = args.names or list(SC.METHODS)
    models, cfgs, stats, params = {}, {}, None, {}
    for name in names:
        # LONGEST matching prefix, not the first: 'flower' is a prefix of
        # 'flowersno', and 'sno' of 'snofast', so first-match would rebuild a
        # Flower+SNO checkpoint as a plain Flower and die on a shape mismatch.
        key  = max((m for m in SC.METHODS if name.startswith(m)),
                   key=len, default=None)
        ckpt = find_best_ckpt(ckpt_dir, name)
        if key is None:
            print(f"  [{name}] cannot infer architecture from name — skipped")
            continue
        if ckpt is None:
            print(f"  [{name}] no checkpoint found — skipped")
            continue
        cfg = dict(base_cfg, _key=key)
        model, used, ck_stats = load_operator(ckpt, cfg, n_in, device)
        models[name] = model
        cfgs[name]   = used
        params[name] = sum(q.numel() for q in model.parameters())
        stats        = ck_stats or stats
        print(f"  [{name}] {ckpt.name}  "
              f"{sum(q.numel() for q in model.parameters()):,} params  "
              f"{SC.describe_model(key, used, used['_n_in'])}")

    if not models:
        print("\nNo checkpoints loaded — nothing to evaluate.")
        return

    if stats is None:
        print("\nno stats in checkpoint; recomputing from the train split")
        stats = load_or_compute_stats(root, split['train'], args.task,
                                      out_dir / f'stats_{args.task}.json')

    ds = ShockDataset(root, split['test'], task=args.task, target=args.target,
                      stats=stats, patch=0, random_crop=False)

    print(f"\nFull-frame evaluation on {ds.n_frames} test frames...")
    results = evaluate(models, cfgs, ds, device, max_frames=args.max_frames)

    methods = [BASELINE_KEY] + list(models.keys())
    agg     = aggregate(results, methods)
    # rank by the headline metric so the comparison reads top-down; the
    # baseline stays pinned first as the reference row.
    ranked  = [BASELINE_KEY] + sorted((m for m in methods if m != BASELINE_KEY),
                                      key=lambda m: agg[m]['rl2'])
    print_table(agg, ranked, params)
    save_csv(results, ranked, out_dir / f'shock_test_{args.task}.csv')
    with open(out_dir / f'shock_test_{args.task}_summary.json', 'w') as f:
        json.dump(agg, f, indent=2)

    print("Figures...")
    plot_summary(results, ranked, fig_dir)
    plot_spectra(results, ranked, fig_dir)
    # Hardest frames by the model-independent baseline, so the selection does
    # not favour or punish any particular method.
    worst = sorted(results, key=lambda r: -r['metrics'][BASELINE_KEY]['rl2'])
    for r in worst[:args.n_frame_figs]:
        plot_frame(r, ranked, fig_dir)
    print(f"  → {args.n_frame_figs} per-frame figures in {fig_dir}")
    print(f"\nDone.  results → {out_dir}   figures → {fig_dir}")


if __name__ == '__main__':
    main()
