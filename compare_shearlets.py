"""
compare_shearlets.py
====================
Compares our self-contained shearlet filter bank (in sno.py) against the
PyShearlets FFST reference implementation, with full visualisation.

Produces four figures:
  1. Filter bank structure  — all Psi[:,:,k] as heatmaps (fftshifted)
  2. Per-filter pixel diff  — |ours - FFST| for every subband
  3. Tight frame partition  — sum(Psi^2, axis=-1) for both implementations
  4. Forward/inverse demo   — apply transform to a test image, compare reconstructions

Run in a notebook cell:
    %run compare_shearlets.py

Or as a script:
    python compare_shearlets.py --H 64 --W 64 --pyshearlets /content/PyShearlets
"""

import sys
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')          # change to 'inline' in a notebook
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  Load FFST (fix Python 3.12 + numpy 2.0 issues automatically)
# ─────────────────────────────────────────────────────────────────────────────

def load_ffst(pyshearlets_dir: str = '/content/PyShearlets'):
    """
    Add PyShearlets to sys.path and fix compatibility issues.
    Returns (scalesShearsAndSpectra, shearletTransformSpect,
             inverseShearletTransformSpect) or raises ImportError.
    """
    p = Path(pyshearlets_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"PyShearlets not found at {pyshearlets_dir}.\n"
            f"Download from https://github.com/grlee77/PyShearlets and set "
            f"--pyshearlets to its path.")

    # Fix Python 3.12 configparser issue in __init__.py
    init_file = p / 'FFST' / '__init__.py'
    if init_file.exists():
        src = init_file.read_text()
        fixed = (src
                 .replace('from numpy.testing import Tester\n', '')
                 .replace('test = Tester().bench\n', '')
                 .replace('bench = Tester().bench\n', ''))
        init_file.write_text(fixed)

    # Fix numpy 2.0: np.NaN removed, np.int removed
    for fname in (p / 'FFST').glob('*.py'):
        src = fname.read_text()
        if 'np.NaN' in src or 'np.int)' in src:
            fname.write_text(
                src.replace('np.NaN', 'np.nan')
                   .replace('.astype(np.int)', '.astype(int)'))

    sys.path.insert(0, str(p))
    from FFST import (scalesShearsAndSpectra,
                      shearletTransformSpect,
                      inverseShearletTransformSpect)
    return scalesShearsAndSpectra, shearletTransformSpect, inverseShearletTransformSpect


# ─────────────────────────────────────────────────────────────────────────────
#  Our self-contained filter bank (copied from sno.py)
# ─────────────────────────────────────────────────────────────────────────────

def _meyeraux(x):
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)
    y[mask] = np.polyval([-20, 70, -84, 35, 0, 0, 0, 0], x[mask])
    y[x > 1] = 1.0
    return y

def _meyerScaling(x):
    x = np.asarray(x, dtype=np.float64); xa = np.abs(x)
    y = np.zeros_like(xa)
    y[xa < 0.5] = 1.0
    mask = (xa >= 0.5) & (xa < 1.0)
    y[mask] = np.cos(np.pi / 2 * _meyeraux(2 * xa[mask] - 1))
    return y

def _meyerWavelet(x):
    def _h(t):
        ta = np.abs(t); out = np.zeros_like(ta)
        out[(ta >= 1) & (ta < 2)] = np.sin(np.pi/2 * _meyeraux(ta[(ta>=1)&(ta<2)] - 1))
        out[(ta >= 2) & (ta < 4)] = np.cos(np.pi/2 * _meyeraux(ta[(ta>=2)&(ta<4)]/2 - 1))
        return out
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(_h(x)**2 + _h(2*x)**2)

def _bump(x):
    x = np.asarray(x, dtype=np.float64)
    val = _meyeraux(1+x)*(x<=0) + _meyeraux(1-x)*(x>0)
    return np.sqrt(np.maximum(val, 0.0))

def _shearlet_spectrum(xi_x, xi_y, a, s):
    y_new = s*np.sqrt(a)*xi_x + np.sqrt(a)*xi_y
    x_new = a*xi_x
    xx = np.where(np.abs(x_new) == 0.0, 1.0, x_new)
    return _meyerWavelet(x_new) * _bump(y_new / xx)

def build_shearlet_filters(H, W, numOfScales=None):
    if numOfScales is None:
        numOfScales = int(np.floor(0.5 * np.log2(max(H, W))))
    shape = np.array([H, W], dtype=int)
    shapem = (np.mod(shape, 2) == 0)
    shape_work = shape.copy(); shape_work[shapem] += 1
    X_max = 2**(2*(numOfScales-1)+1)
    def _axis(n):
        h = np.linspace(0, X_max, (n+1)//2)
        return np.concatenate((-h[-1:0:-1], h))
    xi_x, xi_y = np.meshgrid(_axis(shape_work[1]),
                              _axis(shape_work[0])[::-1], indexing='xy')
    C_hor = np.abs(xi_x) >= np.abs(xi_y); C_ver = ~C_hor
    shearsPerScale = 2**(np.arange(numOfScales)+2)
    n_sh = 1 + int(shearsPerScale.sum())
    Psi  = np.zeros(tuple(shape_work)+(n_sh,), dtype=np.float64)
    Psi[:,:,0] = _meyerScaling(xi_x)*C_hor + _meyerScaling(xi_y)*C_ver
    for j in range(numOfScales):
        a=2**(-2*j); idx=2**j; start=1+int(shearsPerScale[:j].sum()); shift=1
        for k in range(-2**j, 2**j+1):
            Ph=_shearlet_spectrum(xi_x,xi_y,a,k*2**(-j)); Pv=np.rot90(Ph,2).T
            if k==-2**j:   Psi[:,:,start+idx]=Ph*C_hor+Pv*C_ver
            elif k==2**j:  Psi[:,:,start+idx+shift]=Ph*C_hor+Pv*C_ver
            else:
                np_=int(np.mod(idx+1-shift,shearsPerScale[j]))-1
                if np_==-1: np_=int(shearsPerScale[j])-1
                Psi[:,:,start+np_]=Ph; Psi[:,:,start+idx+shift]=Pv; shift+=1
    Psi = Psi[:H,:W,:]
    if shapem[0] or shapem[1]:
        if_=1+int(shearsPerScale[:-1].sum()); half=int((if_+1)/2)
        rel=np.concatenate([np.arange(1,half+1),np.arange(half+2,shearsPerScale[-1])])
        ai=(if_+rel).astype(int)
        if shapem[0]: Psi[0,1:W,ai]=(1/np.sqrt(2))*(Psi[0,1:W,ai]+Psi[0,W-1:0:-1,ai])
        if shapem[1]: Psi[1:H,0,ai]=(1/np.sqrt(2))*(Psi[1:H,0,ai]+Psi[H-1:0:-1,0,ai])
    return np.fft.ifftshift(Psi, axes=(0,1))


# ─────────────────────────────────────────────────────────────────────────────
#  Numerical comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_numerically(H, W, Psi_ours, Psi_ffst):
    """Print a detailed numerical comparison table."""
    print(f"\n{'='*60}")
    print(f"Numerical comparison  H={H}, W={W}")
    print(f"{'='*60}")
    print(f"  Shape:  ours={Psi_ours.shape}  ffst={Psi_ffst.shape}")
    print(f"  dtype:  ours={Psi_ours.dtype}   ffst={Psi_ffst.dtype}")
    print(f"  range:  ours=[{Psi_ours.min():.4f},{Psi_ours.max():.4f}]  "
          f"ffst=[{Psi_ffst.min():.4f},{Psi_ffst.max():.4f}]")
    print(f"  real:   ours={np.isrealobj(Psi_ours)}   ffst={np.isrealobj(Psi_ffst)}")
    print(f"  nonneg: ours={(Psi_ours>=0).all()}  ffst={(Psi_ffst>=0).all()}")

    tight_o = np.max(np.abs(1 - (Psi_ours**2).sum(-1)))
    tight_f = np.max(np.abs(1 - (Psi_ffst**2).sum(-1)))
    print(f"\n  Tight frame max|1-sum(Psi^2)|:")
    print(f"    ours  = {tight_o:.2e}")
    print(f"    FFST  = {tight_f:.2e}")

    diff = np.abs(Psi_ours - Psi_ffst)
    print(f"\n  Pixel difference |ours - FFST|:")
    print(f"    max   = {diff.max():.2e}")
    print(f"    mean  = {diff.mean():.2e}")
    print(f"    median= {np.median(diff):.2e}")
    print(f"    IDENTICAL: {diff.max() < 1e-12}")

    print(f"\n  Per-subband max pixel diff:")
    for k in range(Psi_ours.shape[2]):
        d = np.abs(Psi_ours[:,:,k] - Psi_ffst[:,:,k]).max()
        if d > 1e-14:
            print(f"    k={k:3d}: {d:.2e}  ← non-zero")
        else:
            print(f"    k={k:3d}: {d:.2e}")

    # Roundtrip test
    rng = np.random.default_rng(42)
    f = rng.standard_normal((H, W))
    for label, Psi in [('ours', Psi_ours), ('FFST', Psi_ffst)]:
        ST = np.fft.ifft2(np.fft.fft2(f)[:,:,None]*Psi, axes=(0,1)).real
        fr = np.fft.ifft2((np.fft.fft2(ST,axes=(0,1))*Psi).sum(-1)).real
        print(f"\n  Roundtrip error ({label}): {np.max(np.abs(f-fr)):.2e}")

    return diff


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

BG   = '#0d1117'
AX   = '#161b22'
TEXT = '#e6edf3'
MUT  = '#8b949e'
GRID = '#21262d'

def ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX)
    ax.tick_params(colors=MUT, labelsize=7)
    ax.spines[:].set_color(GRID)
    if title:  ax.set_title(title, color=TEXT, fontsize=8, pad=3, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, color=MUT, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color=MUT, fontsize=7)


def show_filter(ax, psi_k, title='', cmap='inferno'):
    """Display one filter Psi[:,:,k] fftshifted (DC at centre)."""
    ax.imshow(np.fft.fftshift(psi_k), cmap=cmap, vmin=0, vmax=1,
              interpolation='nearest', aspect='equal')
    ax.set_title(title, color=TEXT, fontsize=7, pad=2)
    ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — Filter bank overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_filter_bank(Psi_ours, Psi_ffst, fig_dir, H, W):
    """
    Show all filters side by side: ours (top) vs FFST (bottom).
    Each column = one shearlet k.
    """
    n_sh    = Psi_ours.shape[2]
    n_cols  = n_sh
    n_rows  = 3     # ours / FFST / diff

    fig = plt.figure(figsize=(n_cols * 1.3, n_rows * 1.5 + 0.6))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                             hspace=0.08, wspace=0.04,
                             left=0.04, right=0.98,
                             top=0.92, bottom=0.02)

    for k in range(n_sh):
        ax = fig.add_subplot(gs[0, k])
        show_filter(ax, Psi_ours[:,:,k],
                    title=f'k={k}' if k == 0 else f'{k}')
        if k == 0:
            ax.set_ylabel('Ours', color='#3fb950', fontsize=8, fontweight='bold')

        ax = fig.add_subplot(gs[1, k])
        show_filter(ax, Psi_ffst[:,:,k])
        if k == 0:
            ax.set_ylabel('FFST', color='#58a6ff', fontsize=8, fontweight='bold')
            ax.yaxis.set_label_position('left')
            ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_color(GRID)

        ax = fig.add_subplot(gs[2, k])
        diff = np.abs(Psi_ours[:,:,k] - Psi_ffst[:,:,k])
        im = ax.imshow(np.fft.fftshift(diff), cmap='hot',
                       vmin=0, vmax=1e-12, interpolation='nearest',
                       aspect='equal')
        ax.axis('off')
        if k == 0:
            ax.set_ylabel('|diff|', color='#f78166', fontsize=8, fontweight='bold')

    fig.suptitle(
        f'All {n_sh} shearlet filters  ({H}×{W})\n'
        f'Top: ours | Middle: FFST | Bottom: |difference| (scale 0–1e-12)',
        color=TEXT, fontsize=9, fontweight='bold', y=0.97)

    path = Path(fig_dir) / f'fig1_filter_bank_{H}x{W}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Tight frame partition of unity
# ─────────────────────────────────────────────────────────────────────────────

def plot_tight_frame(Psi_ours, Psi_ffst, fig_dir, H, W):
    """
    Visualise sum_k Psi[:,:,k]^2. Should be 1 everywhere for tight frame.
    Shows:
      - sum(Psi^2) for ours and FFST
      - deviation from 1 for both
      - individual squared filters Psi_k^2 stacked as a bar chart per pixel-row
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                             hspace=0.35, wspace=0.25,
                             left=0.06, right=0.97,
                             top=0.92, bottom=0.06)

    # Sum Psi^2
    sum_o = (Psi_ours**2).sum(-1)
    sum_f = (Psi_ffst**2).sum(-1)

    for col, (label, S, col_c) in enumerate([
            ('Ours: sum(Psi²,-1)', sum_o, '#3fb950'),
            ('FFST: sum(Psi²,-1)', sum_f, '#58a6ff')]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(np.fft.fftshift(S), cmap='RdBu_r',
                       vmin=0.999, vmax=1.001, interpolation='nearest',
                       aspect='equal')
        ax.set_title(label, color=col_c, fontsize=8.5, pad=3, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
            labelcolor=MUT, labelsize=6)

    # Deviation |sum - 1|
    for col, (label, S, col_c) in enumerate([
            ('|ours - 1|', np.abs(sum_o - 1), '#3fb950'),
            ('|FFST - 1|', np.abs(sum_f - 1), '#58a6ff')]):
        ax = fig.add_subplot(gs[0, col+2])
        im = ax.imshow(np.fft.fftshift(S), cmap='hot',
                       vmin=0, vmax=1e-13, interpolation='nearest',
                       aspect='equal')
        ax.set_title(f'{label}  (max={S.max():.1e})',
                     color=col_c, fontsize=8.5, pad=3, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
            labelcolor=MUT, labelsize=6)

    # Per-subband energy along centre row (H//2 in fftshifted = row 0 in our Psi)
    ax = fig.add_subplot(gs[1, :2])
    ax.set_facecolor(AX)
    row_idx = 0   # DC row (after ifftshift)
    # Take a slice along columns for both implementations
    x_cols = np.arange(W)
    bottom = np.zeros(W)
    cmap   = plt.cm.plasma
    n_sh   = Psi_ours.shape[2]
    for k in range(n_sh):
        vals = Psi_ours[row_idx, :, k]**2
        ax.fill_between(x_cols, bottom, bottom + vals,
                        color=cmap(k / n_sh), alpha=0.8)
        bottom += vals
    ax_style(ax, title='Ours: per-subband Psi_k² stacked (DC row)',
             xlabel='Column (freq)', ylabel='Cumulative Psi²')
    ax.axhline(1.0, color='white', lw=0.8, ls='--', label='tight frame')
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    ax = fig.add_subplot(gs[1, 2:])
    ax.set_facecolor(AX)
    bottom = np.zeros(W)
    for k in range(n_sh):
        vals = Psi_ffst[row_idx, :, k]**2
        ax.fill_between(x_cols, bottom, bottom + vals,
                        color=cmap(k / n_sh), alpha=0.8)
        bottom += vals
    ax_style(ax, title='FFST: per-subband Psi_k² stacked (DC row)',
             xlabel='Column (freq)', ylabel='Cumulative Psi²')
    ax.axhline(1.0, color='white', lw=0.8, ls='--', label='tight frame')
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    fig.suptitle(f'Tight Frame Partition of Unity  ({H}×{W})\n'
                  'sum_k Psi_k² should equal 1 everywhere',
                  color=TEXT, fontsize=10, fontweight='bold', y=0.97)

    path = Path(fig_dir) / f'fig2_tight_frame_{H}x{W}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Forward transform coefficients on a test image
# ─────────────────────────────────────────────────────────────────────────────

def plot_coefficients(Psi_ours, Psi_ffst, fig_dir, H, W):
    """
    Apply forward transform to a test image (step function with edges).
    Show:
      - original image
      - per-scale mean coefficient energy (ours vs FFST)
      - spatial coefficient maps for k=0 (low-pass), k=1 (j=0), k=5 (j=1)
      - |diff of coefficients| ours vs FFST
    """
    # Test image: bent ridge (has clear anisotropic edges)
    x_ax = np.linspace(0, 1, W); y_ax = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x_ax, y_ax)
    f = np.where(np.cos(np.pi*X) + 0.4*np.sin(2*np.pi*Y) > 0, 1.0, -1.0)
    f = f + 0.2 * np.random.default_rng(0).standard_normal((H,W))
    f = f.astype(np.float64)

    # Forward transforms
    Ff = np.fft.fft2(f)
    ST_o = np.fft.ifft2(Ff[:,:,None] * Psi_ours, axes=(0,1)).real  # (H,W,n_sh)
    ST_f = np.fft.ifft2(Ff[:,:,None] * Psi_ffst, axes=(0,1)).real

    n_sh   = Psi_ours.shape[2]
    show_k = [0, 1, min(5, n_sh-1), min(13, n_sh-1)]  # lp, j=0, j=1, j=2

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, len(show_k)+2, figure=fig,
                             hspace=0.28, wspace=0.08,
                             left=0.05, right=0.98,
                             top=0.92, bottom=0.05)

    # Row 0: original + selected coefficient maps (ours)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(f, cmap='RdBu_r', interpolation='nearest', aspect='equal')
    ax.set_title('Test image', color=TEXT, fontsize=8, pad=2, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(AX)
    # Energy per scale bar chart
    numOfScales = int(np.floor(0.5*np.log2(max(H,W))))
    shearsPerScale = 2**(np.arange(numOfScales)+2)
    labels, e_ours, e_ffst = ['lp'], [ST_o[:,:,0].var()], [ST_f[:,:,0].var()]
    idx = 1
    for j, ns in enumerate(shearsPerScale):
        e_o = np.mean([ST_o[:,:,idx+i].var() for i in range(ns)])
        e_f = np.mean([ST_f[:,:,idx+i].var() for i in range(ns)])
        e_ours.append(e_o); e_ffst.append(e_f)
        labels.append(f'j={j}'); idx += ns
    x = np.arange(len(labels))
    ax.bar(x-0.2, e_ours, 0.38, color='#3fb950', alpha=0.85, label='Ours')
    ax.bar(x+0.2, e_ffst, 0.38, color='#58a6ff', alpha=0.85, label='FFST')
    ax.set_xticks(x); ax.set_xticklabels(labels, color=MUT, fontsize=7)
    ax_style(ax, title='Coeff energy per scale', ylabel='Var(ST_k)')
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=AX, edgecolor=GRID)

    for col, k in enumerate(show_k):
        lim = max(abs(ST_o[:,:,k].min()), abs(ST_o[:,:,k].max()))

        ax = fig.add_subplot(gs[1, col])
        ax.imshow(ST_o[:,:,k], cmap='RdBu_r', vmin=-lim, vmax=lim,
                  interpolation='nearest', aspect='equal')
        ax.set_title(f'Ours  k={k}', color='#3fb950', fontsize=7.5, pad=2)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, col])
        ax.imshow(ST_f[:,:,k], cmap='RdBu_r', vmin=-lim, vmax=lim,
                  interpolation='nearest', aspect='equal')
        ax.set_title(f'FFST  k={k}', color='#58a6ff', fontsize=7.5, pad=2)
        ax.axis('off')

    # Last column: |diff of ST|
    ax = fig.add_subplot(gs[1, len(show_k)])
    diff_st = np.abs(ST_o - ST_f).mean(-1)
    im = ax.imshow(diff_st, cmap='hot', interpolation='nearest', aspect='equal')
    ax.set_title(f'mean|ST_ours-ST_ffst|\nmax={diff_st.max():.1e}',
                 color='#f78166', fontsize=7.5, pad=2)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02).ax.tick_params(
        labelcolor=MUT, labelsize=6)

    # Reconstruction comparison
    ax = fig.add_subplot(gs[2, len(show_k)])
    xr_o = np.fft.ifft2((np.fft.fft2(ST_o,axes=(0,1))*Psi_ours).sum(-1)).real
    xr_f = np.fft.ifft2((np.fft.fft2(ST_f,axes=(0,1))*Psi_ffst).sum(-1)).real
    diff_rec = np.abs(xr_o - xr_f)
    im = ax.imshow(diff_rec, cmap='hot', interpolation='nearest', aspect='equal')
    ax.set_title(f'|reconst diff|\nmax={diff_rec.max():.1e}',
                 color='#f78166', fontsize=7.5, pad=2)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02).ax.tick_params(
        labelcolor=MUT, labelsize=6)

    fig.suptitle(f'Shearlet Coefficient Maps  ({H}×{W})\n'
                  'Ours (row 2) vs FFST (row 3) — should be identical',
                  color=TEXT, fontsize=10, fontweight='bold', y=0.97)

    path = Path(fig_dir) / f'fig3_coefficients_{H}x{W}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 — Forward/inverse demo on real image
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_demo(Psi_ours, Psi_ffst, fig_dir, H, W):
    """
    Demonstrate forward and inverse transform on a test image.
    Shows original, reconstruction, and error for both implementations.
    Also shows what happens when you zero out fine-scale coefficients.
    """
    x_ax = np.linspace(0, 1, W); y_ax = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x_ax, y_ax)
    # Cartoon-like image: smooth blobs + sharp edge
    f = (np.exp(-((X-0.3)**2+(Y-0.4)**2)/0.02) -
         0.7*np.exp(-((X-0.7)**2+(Y-0.6)**2)/0.03) +
         np.where(np.abs(X-Y-0.1) < 0.03, 1.0, 0.0))
    f = f.astype(np.float64)

    def forward(psi):
        return np.fft.ifft2(np.fft.fft2(f)[:,:,None]*psi, axes=(0,1)).real
    def inverse(st, psi):
        return np.fft.ifft2((np.fft.fft2(st,axes=(0,1))*psi).sum(-1)).real

    ST_o = forward(Psi_ours); ST_f = forward(Psi_ffst)
    rec_o = inverse(ST_o, Psi_ours); rec_f = inverse(ST_f, Psi_ffst)

    # Coarse-only reconstruction (keep only low-pass + j=0)
    numOfScales = int(np.floor(0.5*np.log2(max(H,W))))
    shearsPerScale = 2**(np.arange(numOfScales)+2)
    n_coarse = 1 + int(shearsPerScale[0])  # lp + j=0
    ST_coarse = ST_o.copy(); ST_coarse[:,:,n_coarse:] = 0
    rec_coarse = inverse(ST_coarse, Psi_ours)

    # Fine-only (remove low-pass and coarsest scale)
    ST_fine = ST_o.copy(); ST_fine[:,:,:n_coarse] = 0
    rec_fine = inverse(ST_fine, Psi_ours)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                             hspace=0.12, wspace=0.06,
                             left=0.04, right=0.98,
                             top=0.92, bottom=0.04)

    items_top = [
        ('Original f',       f,       'RdBu_r'),
        ('Ours: iST(ST(f))', rec_o,   'RdBu_r'),
        ('FFST: iST(ST(f))', rec_f,   'RdBu_r'),
        ('Coarse only\n(lp+j=0)', rec_coarse, 'RdBu_r'),
        ('Fine only\n(j>=1)', rec_fine, 'RdBu_r'),
    ]
    vlim = max(abs(f.min()), abs(f.max()))

    for col, (title, data, cmap) in enumerate(items_top):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(data, cmap=cmap, vmin=-vlim, vmax=vlim,
                  interpolation='nearest', aspect='equal')
        ax.set_title(title, color=TEXT, fontsize=8, pad=2, fontweight='bold')
        ax.axis('off')

    items_bot = [
        ('',                    None,                   None),
        ('|f - rec_ours|',      np.abs(f-rec_o),        'hot'),
        ('|f - rec_ffst|',      np.abs(f-rec_f),        'hot'),
        ('|f - coarse|',        np.abs(f-rec_coarse),   'hot'),
        ('|f - fine|',          np.abs(f-rec_fine),     'hot'),
    ]
    err_max = max(np.abs(f-rec_o).max(), np.abs(f-rec_f).max(),
                  np.abs(f-rec_coarse).max(), np.abs(f-rec_fine).max())

    for col, (title, data, cmap) in enumerate(items_bot):
        ax = fig.add_subplot(gs[1, col])
        if data is None:
            ax.set_facecolor(BG); ax.axis('off')
            ax.text(0.5, 0.5, f'max err (ours): {np.abs(f-rec_o).max():.2e}\n'
                    f'max err (FFST): {np.abs(f-rec_f).max():.2e}',
                    ha='center', va='center', color=TEXT, fontsize=8,
                    transform=ax.transAxes)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=err_max,
                      interpolation='nearest', aspect='equal')
            ax.set_title(title, color='#f78166', fontsize=8, pad=2)
            ax.axis('off')

    fig.suptitle('Forward + Inverse Shearlet Transform Demo\n'
                  'Coarse = low-freq structure  |  Fine = edges and detail',
                  color=TEXT, fontsize=10, fontweight='bold', y=0.97)

    path = Path(fig_dir) / f'fig4_reconstruction_{H}x{W}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=64)
    p.add_argument('--W', type=int, default=64)
    p.add_argument('--pyshearlets', default='/content/PyShearlets',
                   help='Path to PyShearlets root directory')
    p.add_argument('--fig_dir', default='figures/shearlet_compare')
    args = p.parse_args()

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    H, W = args.H, args.W

    # Load FFST
    print(f"Loading FFST from {args.pyshearlets}...")
    try:
        scalesShearsAndSpectra, _, _ = load_ffst(args.pyshearlets)
        Psi_ffst = scalesShearsAndSpectra((H, W))
        print(f"  FFST loaded: Psi shape = {Psi_ffst.shape}")
    except (FileNotFoundError, ImportError) as e:
        print(f"  WARNING: {e}\n  Running ours-only mode.")
        Psi_ffst = None

    # Build ours
    print(f"\nBuilding our filter bank ({H}x{W})...")
    Psi_ours = build_shearlet_filters(H, W)
    tight = np.max(np.abs(1 - (Psi_ours**2).sum(-1)))
    print(f"  n_sh={Psi_ours.shape[2]}  tight_frame_err={tight:.2e}")

    if Psi_ffst is not None:
        diff = compare_numerically(H, W, Psi_ours, Psi_ffst)
    else:
        Psi_ffst = Psi_ours   # fallback: show ours twice

    print(f"\nGenerating figures in {args.fig_dir}/")
    plot_filter_bank(Psi_ours, Psi_ffst, args.fig_dir, H, W)
    plot_tight_frame(Psi_ours, Psi_ffst, args.fig_dir, H, W)
    plot_coefficients(Psi_ours, Psi_ffst, args.fig_dir, H, W)
    plot_reconstruction_demo(Psi_ours, Psi_ffst, args.fig_dir, H, W)

    print("\nDone. Figures:")
    for f in sorted(Path(args.fig_dir).glob('fig*.png')):
        print(f"  {f}")


if __name__ == '__main__':
    main()
