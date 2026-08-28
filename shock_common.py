"""
shock_common.py — model factory, seam handling, metrics, full-frame inference
==============================================================================
Everything shared by `train_shock.py` and `test_shock.py` that does NOT need
PyTorch Lightning, so it can be unit-tested standalone.

THE PERIODICITY PROBLEM
-----------------------
All four operators (FNO, SNO, USNO, CascadeUSNO) are FFT-based and therefore
treat the domain as periodic.  These fields are not: the walls are reflecting
and the driver gas occupies x < 0.1, so wrapping x=1 back to x=0 creates a
discontinuity measured at ~100x (rho) and ~139x (E) the mean interior
gradient.  Left unhandled, every model burns capacity on Gibbs ringing at a
seam that is a pure artifact.

`PadWrap` fixes it by reflect-padding the input before the operator and
cropping the result afterwards, so the wrap-around the FFT sees is a mirror
of the interior rather than the opposite wall.  The operator is built for the
PADDED size, which is why `padded_size()` must be used when constructing it.

FAIR COMPARISON: FNO BANDWIDTH
------------------------------
`matched_k_max(n_sh)` gives k_max = round(sqrt(n_sh/2)) -- 4 at 128-129, 6 at
257 -- which equalises the SPECTRAL PARAMETER COUNT against SNO.  But it also
leaves FNO with a 4x4 block of Fourier modes out of 128, which cannot resolve
a shock front; FNO's only other path is a pointwise 1x1 conv that does no
spatial mixing at all.  A "SNO beats FNO on fine detail" result obtained only
at matched k_max is attributable to bandwidth, not to shearlet anisotropy.

Always run the bandwidth control too:

    --methods fno --k_max 32 --tag fno_wide     # more params than SNO
    --methods fno                               # param-matched, k_max=4

If SNO still wins against wide-band FNO, the result is about the basis.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fno_urban          import FNO, matched_k_max
from sno_urban          import SNO, build_shearlet_filters  # SNO: reference impl
from usno_urban         import USNO
from cascade_usno_urban import CascadeUSNO
from warp_sno           import WarpSNO
from sno_fast            import FastSNO, FlowerSNO

METHODS = ('fno', 'sno', 'usno', 'cascade', 'flower', 'warpsno',
           'snofast', 'flowersno')
LABELS  = {'fno': 'FNO', 'sno': 'SNO',
           'usno': 'USNO', 'cascade': 'Cascade USNO',
           'flower': 'Flower', 'warpsno': 'WarpSNO',
           'snofast': 'SNO (alias)', 'flowersno': 'Flower+SNO'}
# Eight series need eight separable hues.  The earlier palette had sno and
# snofast identical, fno/flower both blue and usno/flowersno both green, which
# is unreadable in the spectrum plot where the curves overlap.
COLORS  = {'fno':      '#58a6ff',   # blue
           'sno':      '#f78166',   # salmon
           'usno':     '#3fb950',   # green
           'cascade':  '#d2a8ff',   # lavender
           'flower':   '#ffd33d',   # yellow
           'warpsno':  '#ff7b72',   # red
           'snofast':  '#8957e5',   # violet
           'flowersno':'#39d3c3'}   # teal

# Param-matched Flower baseline.  lifting_dim=28 / n_levels=3 / num_heads=7
# gives 132,206 params at 160², within 6% of SNO (124,356) and 3% of WarpSNO
# (135,684).  Flower's own default (160/4/40) is 17.3M — two orders of
# magnitude past what 322 training frames support; cf. fno_wide at 16.8M,
# which collapsed to skill 0.017.
FLOWER_DEFAULTS = dict(lifting_dim=28, n_levels=3, num_heads=7, groups=7)


def _opt(cfg: dict, key: str, default):
    """cfg.get(key, default) that also treats an explicit None as unset.

    argparse writes None for every flag the user did not pass, so the key is
    present and dict.get's default is never used."""
    v = cfg.get(key)
    return default if v is None else v


def resolve_cfg(key: str, cfg: dict) -> dict:
    """
    Copy of cfg with this method's defaults filled in.

    Unset CLI flags arrive as None, so a checkpoint trained with defaults would
    record `lifting_dim: None` and rebuild against whatever FLOWER_DEFAULTS
    happens to say at evaluation time.  Resolving here makes every checkpoint
    self-describing.
    """
    out = dict(cfg)
    if key == 'flowersno':
        for k, v in FLOWER_DEFAULTS.items():
            out[k] = _opt(cfg, k, v)
        out['sno_at']       = _opt(cfg, 'sno_at', 'head')
        out['sno_channels'] = _opt(cfg, 'sno_channels', out['lifting_dim'])
    elif key == 'flower':
        for k, v in FLOWER_DEFAULTS.items():
            out[k] = _opt(cfg, k, v)
    elif key == 'warpsno':
        out['num_heads']   = _opt(cfg, 'num_heads', 8)
        out['warp_guided'] = _opt(cfg, 'warp_guided', True)
        out['max_disp']    = _opt(cfg, 'max_disp', 3.0)
        out['warp_energy'] = _opt(cfg, 'warp_energy', 'scale')
    return out


def _import_flower():
    """Lazy import: the vendored Flower needs einops, the others do not."""
    import sys
    from pathlib import Path
    d = str(Path(__file__).resolve().parent / 'flowers')
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        from flower_standalone import Flower
    except ImportError as e:
        raise ImportError(
            "method 'flower' needs the vendored flowers/ checkout and einops "
            f"(pip install einops). Original error: {e}") from None
    return Flower


# ─────────────────────────────────────────────────────────────────────────────
#  Seam handling
# ─────────────────────────────────────────────────────────────────────────────

def padded_size(n: int, pad: int) -> int:
    """Grid size the operator must be built for, given reflect padding."""
    return n + 2 * pad


class PadWrap(nn.Module):
    """
    Reflect-pad -> operator -> crop.

    The wrapped operator sees a (n+2p)x(n+2p) field whose periodic wrap is a
    mirror of the interior instead of the opposite wall.  Output is cropped
    back to nxn, so the module is shape-preserving from the caller's view.

    pad=0 disables padding entirely (the operator runs on the raw field).
    """

    def __init__(self, model: nn.Module, pad: int = 0):
        super().__init__()
        self.model = model
        self.pad   = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        if p == 0:
            return self.model(x)
        # 'reflect' does not duplicate the edge sample, which is what we want:
        # the mirrored copy continues the field smoothly across the boundary.
        x = F.pad(x, (p, p, p, p), mode='reflect')
        y = self.model(x)
        return y[..., p:-p, p:-p]

    def param_count(self) -> int:
        return sum(q.numel() for q in self.parameters() if q.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(key: str, cfg: dict, n: int) -> PadWrap:
    """
    Build one operator, already wrapped in its PadWrap.

    `n` is the size of the field the caller will hand in (patch size, or the
    full grid).  The operator itself is constructed for n + 2*pad.
    """
    pad   = int(cfg.get('pad', 0))
    n_pad = padded_size(n, pad)
    size  = (n_pad, n_pad)
    C     = cfg['hidden']
    cin   = cfg.get('in_channels', 4)
    cout  = cfg.get('out_channels', 4)

    if key == 'fno':
        n_sh  = build_shearlet_filters(n_pad, n_pad,
                                       numOfScales=cfg.get('n_scales')).shape[2]
        k_max = cfg.get('k_max') or matched_k_max(n_sh)
        # rfft2 keeps floor(W/2)+1 columns; k_max cannot exceed what exists.
        k_max = min(int(k_max), n_pad // 2)
        model = FNO(in_channels=cin, out_channels=cout, hidden_channels=C,
                    k_max=k_max, n_blocks=cfg['n_blocks'])
    elif key in ('sno', 'snofast'):
        # FastSNO is sno_urban.SNO rewritten as a shearlet-parameterised
        # Fourier multiplier: identical outputs (verified to 1e-9 against the
        # trained checkpoint), identical state_dict, ~8x faster.  'snofast' is
        # a deprecated alias kept so checkpoints written under that name stay
        # addressable; new runs should use 'sno'.
        model = FastSNO(in_channels=cin, out_channels=cout, hidden_channels=C,
                        n_blocks=cfg['n_blocks'], n_scales=cfg.get('n_scales'),
                        input_size=size)
    elif key == 'usno':
        model = USNO(in_channels=cin, out_channels=cout, hidden_channels=C,
                     n_scales=cfg.get('n_scales'), n_layers=cfg['n_layers'],
                     input_size=size)
    elif key == 'cascade':
        model = CascadeUSNO(in_channels=cin, out_channels=cout,
                            hidden_channels=C, n_layers=cfg['n_layers'],
                            n_scales=cfg.get('n_scales'), input_size=size)
    elif key == 'flowersno':
        f = {**FLOWER_DEFAULTS,
             **{k: cfg[k] for k in ('lifting_dim', 'n_levels', 'num_heads',
                                    'groups') if cfg.get(k) is not None}}
        model = FlowerSNO(in_channels=cin, out_channels=cout,
                          hidden_channels=_opt(cfg, 'sno_channels', f['lifting_dim']),
                          input_size=size, n_scales=cfg.get('n_scales'),
                          sno_at=_opt(cfg, 'sno_at', 'head'), **f)
    elif key == 'flower':
        # Plain Flower — NO shearlets.  This is the transport-only baseline
        # that isolates how much of the correction is pure warping; 'warpsno'
        # is the hybrid.  Flower's U-Net halves the grid n_levels-1 times, so
        # n_pad must be divisible by 2**(n_levels-1): 160 works (sr, patch
        # 128), 161 cannot (same, 129 is odd -> 129+2p is always odd).
        Flower = _import_flower()
        f = {**FLOWER_DEFAULTS,
             **{k: cfg[k] for k in ('lifting_dim', 'n_levels', 'num_heads',
                                    'groups') if cfg.get(k) is not None}}
        div = 2 ** (f['n_levels'] - 1)
        if n_pad % div:
            raise ValueError(
                f"Flower needs the padded grid divisible by 2**(n_levels-1)"
                f"={div}, but n={n} with pad={pad} gives {n_pad}. Use a patch "
                f"size that makes n+2*pad divisible, or lower --n_levels.")
        model = Flower(dim_in=cin, dim_out=cout, n_spatial_dims=2,
                       spatial_resolution=[n_pad, n_pad],
                       boundary_condition_types=['DIRICHLET', 'DIRICHLET'],
                       dropout_rate=0.0, **f)
    elif key == 'warpsno':
        # NB: cfg carries explicit None for unset CLI flags, so dict.get's
        # default never fires — fall back on the value, not on the key.
        heads = _opt(cfg, 'num_heads', 8)
        if C % heads:
            div = [d for d in range(1, C + 1) if C % d == 0]
            raise ValueError(
                f"warpsno needs hidden ({C}) divisible by num_heads "
                f"({heads}). Divisors of {C}: {div}. Note --num_heads is "
                f"shared with 'flower', whose default (7) does not divide 32 "
                f"— leave it unset to give each method its own default.")
        model = WarpSNO(in_channels=cin, out_channels=cout, hidden_channels=C,
                        n_blocks=cfg['n_blocks'], n_scales=cfg.get('n_scales'),
                        input_size=size, num_heads=heads,
                        guided=_opt(cfg, 'warp_guided', True),
                        max_disp=_opt(cfg, 'max_disp', 3.0),
                        energy=_opt(cfg, 'warp_energy', 'scale'))
    else:
        raise ValueError(f"unknown method {key!r}; expected one of {METHODS}")

    return PadWrap(model, pad)


def describe_model(key: str, cfg: dict, n: int) -> str:
    n_pad = padded_size(n, int(cfg.get('pad', 0)))
    if key in ('flower', 'flowersno'):
        f = {**FLOWER_DEFAULTS,
             **{k: cfg[k] for k in ('lifting_dim', 'n_levels', 'num_heads',
                                    'groups') if cfg.get(k) is not None}}
        if key == 'flower':
            return (f"grid {n}->{n_pad}  lifting_dim={f['lifting_dim']}  "
                    f"n_levels={f['n_levels']}  heads={f['num_heads']}  "
                    f"(no shearlets — transport only)")
        n_sh = build_shearlet_filters(n_pad, n_pad,
                                      numOfScales=cfg.get('n_scales')).shape[2]
        return (f"grid {n}->{n_pad}  lifting_dim={f['lifting_dim']}  "
                f"n_levels={f['n_levels']}  heads={f['num_heads']}  "
                f"n_sh={n_sh}  sno_at={_opt(cfg, 'sno_at', 'head')}")
    n_sh  = build_shearlet_filters(n_pad, n_pad,
                                   numOfScales=cfg.get('n_scales')).shape[2]
    if key == 'fno':
        k = min(int(cfg.get('k_max') or matched_k_max(n_sh)), n_pad // 2)
        return f"grid {n}->{n_pad}  k_max={k}  (param-matched k_max={matched_k_max(n_sh)})"
    return f"grid {n}->{n_pad}  n_sh={n_sh}"


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation helpers  (mirror shock_dataset.ShockDataset)
# ─────────────────────────────────────────────────────────────────────────────

def stats_tensors(stats: Dict[str, Sequence[float]], device=None, dtype=None):
    def _t(k):
        return torch.as_tensor(stats[k], dtype=dtype or torch.float32,
                               device=device).view(1, -1, 1, 1)
    return _t('mean'), _t('std'), _t('res_std')


def denormalise(out: torch.Tensor, x_phys: torch.Tensor,
                stats: Dict[str, Sequence[float]], target: str) -> torch.Tensor:
    """Network output -> physical field.  Inverse of ShockDataset.__getitem__."""
    mean, std, res_std = stats_tensors(stats, out.device, out.dtype)
    if target == 'residual':
        return x_phys + out * res_std
    return mean + out * std


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2, averaged over the batch."""
    d = (pred - target).reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (d.norm(dim=1) / (t.norm(dim=1) + 1e-12)).mean()


def rel_l2_per_channel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    (C,) relative L2 per channel, averaged over the batch.

    A channel that is identically zero in the target has NO defined relative
    error, and dividing by an epsilon manufactures a huge finite number that
    then poisons any mean.  On this data that is not hypothetical: task='sr'
    keeps the t=0 frame, where rho_v == 0 exactly, and the old guard turned
    that into ~1e10 and made the rl2_rho_v column meaningless.  Return NaN for
    those channels and let the callers use nanmean.
    """
    d  = (pred - target).reshape(*pred.shape[:2], -1)
    t  = target.reshape(*target.shape[:2], -1)
    tn = t.norm(dim=2)
    # Threshold on the channel's SHARE of the field norm, not an absolute
    # epsilon.  Measured on the sr test split: rho_v holds 0 of the field at
    # t=0 and 3.6e-7 at t=1 (relative error 2028, pure noise amplification),
    # but 2.6e-3 by t=2 (0.475) and >1e-2 from t=3 on (0.07-0.18).  1e-5 cuts
    # exactly the two degenerate frames and keeps every informative one.
    scale = target.reshape(target.shape[0], -1).norm(dim=1, keepdim=True)
    out   = d.norm(dim=2) / tn.clamp(min=1e-12)
    out   = torch.where(tn > 1e-5 * scale.clamp(min=1e-12), out,
                        torch.full_like(out, float('nan')))
    return out.nanmean(dim=0)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Self-normalising MSE — scale-free, so channel weighting is uniform."""
    return ((pred - target) ** 2).mean() / ((target ** 2).mean().clamp(min=1e-12))


def _radial_mask(h: int, w: int, cut: float, device) -> torch.Tensor:
    """Boolean (h,w) mask selecting radial frequencies above `cut` x Nyquist."""
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=device).view(1, -1)
    return (torch.sqrt(fy ** 2 + fx ** 2) >= 0.5 * cut)


def highpass_rel_l2(pred: torch.Tensor, target: torch.Tensor,
                    cut: float = 0.25) -> torch.Tensor:
    """
    Relative L2 restricted to the high-frequency band.

    Global relL2 on this data is dominated by the large smooth regions, which
    every model gets roughly right; it hides exactly the fine-detail
    difference this study is about.  This keeps only radial frequencies above
    `cut` x Nyquist (default: the top three quarters of the spectrum) and
    measures the error there.
    """
    h, w = pred.shape[-2], pred.shape[-1]
    m    = _radial_mask(h, w, cut, pred.device)
    pf   = torch.fft.fft2(pred)   * m
    tf   = torch.fft.fft2(target) * m
    d    = (pf - tf).reshape(pf.shape[0], -1).abs()
    t    = tf.reshape(tf.shape[0], -1).abs()
    return (d.norm(dim=1) / (t.norm(dim=1) + 1e-12)).mean()


def front_rel_l2(pred: torch.Tensor, target: torch.Tensor,
                 channel: int = 0, quantile: float = 0.90) -> torch.Tensor:
    """
    Relative L2 evaluated only where the shock fronts are.

    The mask is the top (1-quantile) fraction of |grad rho| in the TARGET, so
    it is model-independent: every method is scored on the same pixels.
    """
    t   = target[:, channel]
    gy  = torch.gradient(t, dim=1)[0]
    gx  = torch.gradient(t, dim=2)[0]
    mag = (gy.abs() + gx.abs()).reshape(t.shape[0], -1)

    thr  = torch.quantile(mag, quantile, dim=1, keepdim=True)
    mask = (mag >= thr).view(t.shape[0], 1, *t.shape[1:]).to(pred.dtype)

    d = ((pred - target) * mask).reshape(pred.shape[0], -1)
    n = (target * mask).reshape(target.shape[0], -1)
    return (d.norm(dim=1) / (n.norm(dim=1) + 1e-12)).mean()


def radial_spectrum(field: np.ndarray, n_bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radially averaged power spectrum of a single (H, W) field.

    Used to show WHERE in the spectrum a model puts its energy — a model that
    under-predicts fine detail has a spectrum that falls off too fast.
    """
    f  = np.fft.fft2(field - field.mean())
    p  = np.abs(f) ** 2
    h, w = field.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r  = np.sqrt(fy ** 2 + fx ** 2)

    edges   = np.linspace(0, r.max(), n_bins + 1)
    idx     = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    counts  = np.bincount(idx, minlength=n_bins)
    sums    = np.bincount(idx, weights=p.ravel(), minlength=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, sums / np.maximum(counts, 1)


def mass_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative error in total mass (integral of rho) — a conservation check."""
    p = pred[:, 0].sum(dim=(-2, -1))
    t = target[:, 0].sum(dim=(-2, -1))
    return ((p - t).abs() / t.abs().clamp(min=1e-12)).mean()


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Every scalar metric, on PHYSICAL fields.  pred/target: (B, 4, H, W)."""
    out = {
        'rl2':          rel_l2(pred, target).item(),
        'rl2_highpass': highpass_rel_l2(pred, target).item(),
        'rl2_front':    front_rel_l2(pred, target).item(),
        'mass_err':     mass_error(pred, target).item(),
    }
    per = rel_l2_per_channel(pred, target)
    for i, name in enumerate(('rho', 'rho_u', 'rho_v', 'E')):
        out[f'rl2_{name}'] = per[i].item()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Full-frame inference
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_window(h: int, w: int, device=None) -> torch.Tensor:
    gy = torch.exp(-(torch.linspace(-1, 1, h, device=device) ** 2) / 0.5)
    gx = torch.exp(-(torch.linspace(-1, 1, w, device=device) ** 2) / 0.5)
    return torch.outer(gy, gx)


@torch.no_grad()
def predict_frame(model: nn.Module, x_phys: np.ndarray,
                  stats: Dict[str, Sequence[float]],
                  target: str = 'residual',
                  patch: int = 0,
                  stride: Optional[int] = None,
                  device=None) -> np.ndarray:
    """
    Physical input field -> physical predicted field, (4, N, N) numpy.

    patch = 0  : one forward pass over the whole frame.
    patch = P  : overlapping PxP sliding window, Gaussian-blended.  Required
                 when the operator was built for P (the shearlet filter bank
                 is fixed at construction, so a model trained on patches
                 cannot be run at full resolution).
    """
    if device is None:
        # A model may legitimately have no parameters (baselines, identity),
        # in which case next(...) would raise StopIteration.
        device = next(model.parameters(), torch.zeros(0)).device
    model.eval()

    xt = torch.from_numpy(np.ascontiguousarray(x_phys)).float().unsqueeze(0).to(device)
    mean, std, _ = stats_tensors(stats, device, xt.dtype)
    xn = (xt - mean) / std

    if patch == 0:
        out = model(xn)
    else:
        stride = stride or max(1, patch // 2)
        _, _, H, W = xn.shape
        if patch > H or patch > W:
            raise ValueError(f"patch {patch} exceeds frame {H}x{W}")
        win   = gaussian_window(patch, patch, device).view(1, 1, patch, patch)
        accum = torch.zeros_like(xn)
        count = torch.zeros(1, 1, H, W, device=device, dtype=xn.dtype)

        ys = list(range(0, H - patch + 1, stride))
        xs = list(range(0, W - patch + 1, stride))
        if ys[-1] != H - patch:
            ys.append(H - patch)          # always cover the far edge
        if xs[-1] != W - patch:
            xs.append(W - patch)

        for y in ys:
            for x in xs:
                p = model(xn[:, :, y:y + patch, x:x + patch])
                accum[:, :, y:y + patch, x:x + patch] += p * win
                count[:, :, y:y + patch, x:x + patch] += win
        out = accum / count.clamp(min=1e-8)

    return denormalise(out, xt, stats, target).squeeze(0).cpu().numpy()
