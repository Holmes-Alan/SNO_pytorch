"""
warp_sno.py — WarpSNO: shearlet spectral mixing + Flower-style learned warps
=============================================================================
A hybrid of SNO (`sno_urban.py`) and Flower (`flowers/flower_standalone.py`).

NAMING — the two are separate methods in `shock_common.METHODS`:
    'flower'   plain Flower, no shearlets.  The transport-only baseline that
               isolates how much of the correction is pure warping.
    'warpsno'  this file: shearlet spectral mixing + a Flower-style warp
               branch in every block.
Never report a 'flower' run under the WarpSNO name — it contains no SNO.

WHY COMBINE THEM
----------------
The two architectures fail in complementary ways on shock data:

  SNO      The shearlet frame is near-optimal for cartoon-like functions
           (piecewise smooth with C² discontinuity curves) — exactly a shock
           front.  But `Psi` is a FIXED buffer and the only learned object is
           a per-subband channel matrix W, so the operator is STATIONARY: it
           applies the same spectral multiplier everywhere.  It can sharpen a
           front; it cannot move one.

  Flower   `SelfWarp` predicts a per-pixel displacement and resamples,
           u(x) -> u(x + d(x)).  Transport is native and costs O(1)
           parameters.  But the resampling is bilinear, which is diffusive:
           it moves a front and blurs it on the way.  And the flow head is
           pointwise on raw features — it has no directional/multiscale view
           of the field it is deciding to warp.

MEASURED ON shock_bubble_sr (see the oracle-warp probe)
-------------------------------------------------------
An ORACLE smooth warp — Lucas-Kanade fitted against the target, so it is an
upper bound on any learned flow using the same operator — removes:

                       global rl2    high-pass    at fronts   mean |d|
    task=same             32.7%        26.2%        36.0%     0.90 cells
    task=sr               50.9%        37.3%        55.2%     1.71 cells

Learned SNO, with no oracle access, already removes ~61% / ~64% / ~66%.

Two conclusions, both load-bearing for this design:

  1. A third to a half of the coarse solver's error IS a smooth sub-cell
     displacement of the fronts.  That component is invisible to a stationary
     spectral multiplier, and a warp expresses it with almost no parameters.
  2. The warp is weakest exactly in the high-pass band (26–37%), because a
     resampling cannot restore energy the coarse scheme never resolved.  That
     is the shearlet branch's job.

So: warp for WHERE the front is, shearlets for HOW SHARP it is.  Neither
subsumes the other, and Flower alone would be a downgrade here.

WHAT THIS ADDS OVER A NAIVE CONCATENATION
------------------------------------------
The shearlet analysis is computed anyway for the spectral branch.  Its
per-scale energy is a free, orientation- and scale-aware description of where
the fronts are — precisely what the flow head needs to decide a displacement.
`WarpSNOBlock` computes the analysis ONCE and feeds its per-scale energy to
the flow head, so the warp is shearlet-guided at no extra FFT cost.  This also
respects Flower's design constraint that displacements stay pointwise: the
spatial aggregation lives in the (fixed, non-learned) shearlet filters.

Displacements are tanh-bounded to `max_disp` cells.  The probe says the true
correction is ~1–2 cells; an unbounded flow field on 322 training frames
thrashes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sno_urban  import build_shearlet_filters
from usno_urban import _scale_of_subband


# ─────────────────────────────────────────────────────────────────────────────
#  Shearlet analysis that also reports per-scale energy
# ─────────────────────────────────────────────────────────────────────────────

class ShearletMix(nn.Module):
    """
    Shearlet spectral mixing that ALSO reports per-scale energy for the flow
    head, in the multiplier form (see sno_fast.py for the derivation).

    SYNTHESIS is exact and free.  Every Psi_k is even in frequency and x is
    real, so FFT(ST_k) = Xf . Psi_k and

        out = IFFT( M(xi) @ Xf(xi) ).real,   M(xi) = sum_k W_k Psi_k(xi)^2

    which never materialises the (B, C, n_sh, H, W) coefficient tensor.

    ENERGY is the part that cannot be had for free, because the flow head wants
    |ST|^2 pooled per scale, and that needs the coefficients themselves.  Two
    options:

      energy='scale'   (default) Filter with the scale-j bandpass
                       B_j = sum_{k in j} Psi_k^2, which is a partition of
                       unity over scales (sum_j B_j = 1), and take the energy
                       of each band: J+1 = 4 filtered fields instead of 29.
                       Not identical to sum_k ST_k^2 — it is the energy of the
                       whole scale band rather than the sum of squares of the
                       individual shear responses — but it carries the same
                       scale information the flow head is conditioned on, and
                       it is better conditioned (no dependence on how many
                       shears a scale happens to have).

      energy='subband' The exact sum_k ST_k^2 per scale.  Keeps the full
                       analysis, so it costs ~7x more than 'scale', but still
                       saves the synthesis FFT and the `mixed` tensor versus
                       the original implementation.

    'scale' changes the model, so a checkpoint trained under one setting is not
    valid under the other.
    """

    def __init__(self, channels: int, Psi: torch.Tensor,
                 scale_of: np.ndarray, energy: str = 'scale'):
        super().__init__()
        if energy not in ('scale', 'subband'):
            raise ValueError(f"energy must be scale|subband, got {energy!r}")
        self.C      = channels
        self.energy = energy
        n_sh        = int(Psi.shape[0])
        self.register_buffer('Psi', Psi.float())                  # (n_sh,H,W)

        scale  = 1.0 / (channels * channels)
        self.W = nn.Parameter(scale * torch.randn(n_sh, channels, channels))

        J   = int(scale_of.max())
        sel = torch.zeros(J + 1, n_sh)
        for k, j in enumerate(scale_of):
            sel[int(j), k] = 1.0
        self.register_buffer('sel', sel)                          # (J+1,n_sh)
        self.n_scales_out = J + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W  = x.shape[-2], x.shape[-1]
        Wh    = W // 2 + 1
        Psi_h = self.Psi[..., :Wh]                                # (n_sh,H,Wh)
        G     = Psi_h ** 2
        Xf    = torch.fft.rfft2(x)                                # (B,C,H,Wh)

        # ── per-scale energy, for the flow head ──────────────────────────────
        if self.energy == 'scale':
            Bj  = torch.einsum('jk,khw->jhw', self.sel, G)        # (J+1,H,Wh)
            u   = torch.fft.irfft2(Xf.unsqueeze(2) * Bj.sqrt(), s=(H, W))
            eng = u.pow(2).mean(dim=1)                            # (B,J+1,H,W)
        else:
            ST  = torch.fft.irfft2(Xf.unsqueeze(2) * Psi_h, s=(H, W))
            e   = ST.pow(2).mean(dim=1)                           # (B,n_sh,H,W)
            eng = torch.einsum('jk,bkhw->bjhw', self.sel, e)
        eng = eng.clamp(min=1e-12).sqrt()

        # ── synthesis, exact multiplier form ─────────────────────────────────
        M  = torch.einsum('koi,khw->oihw', self.W, G)             # (C,C,H,Wh)
        re = torch.einsum('oihw,bihw->bohw', M, Xf.real)
        im = torch.einsum('oihw,bihw->bohw', M, Xf.imag)
        out = torch.fft.irfft2(torch.complex(re, im), s=(H, W))
        return out, eng


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-head warp  (Flower's SelfWarp, bounded and shearlet-guided)
# ─────────────────────────────────────────────────────────────────────────────

class SelfWarp2d(nn.Module):
    """
    u(x) -> u(x + d(x)), multi-head, with a pointwise flow head.

    Differences from `flowers.SelfWarp`, each forced by this problem:
      * displacement is tanh-bounded to `max_disp` CELLS (probe: ~1-2 cells)
      * the flow head may be conditioned on shearlet per-scale energy
      * padding_mode='border': PadWrap already reflect-pads the domain, so the
        warp must not wrap across the reflecting walls
    """

    def __init__(self, channels: int, size: Tuple[int, int],
                 num_heads: int = 8, guide_dim: int = 0,
                 max_disp: float = 3.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels {channels} not divisible by "
                             f"num_heads {num_heads}")
        self.C, self.heads, self.max_disp = channels, num_heads, float(max_disp)
        H, W = size

        self.flow_head = nn.Sequential(
            nn.Conv2d(channels + guide_dim, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2 * num_heads, 1))
        self.value_head = nn.Conv2d(channels, channels, 1)

        # zero-init the last flow layer: the block starts as pure identity
        # transport, so it cannot destabilise early training.
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H),
                                torch.linspace(-1, 1, W), indexing='ij')
        self.register_buffer('base', torch.stack([xx, yy], -1)[None])  # (1,H,W,2)
        self.register_buffer('cell', torch.tensor([2.0 / max(W - 1, 1),
                                                   2.0 / max(H - 1, 1)]))

    def forward(self, x: torch.Tensor,
                guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        f_in = x if guide is None else torch.cat([x, guide], dim=1)

        # (B, 2*heads, H, W) -> (B*heads, H, W, 2), displacement in cells
        d = torch.tanh(self.flow_head(f_in)) * self.max_disp
        d = d.view(B * self.heads, 2, H, W).permute(0, 2, 3, 1)
        grid = self.base + d * self.cell                    # broadcast (1,H,W,2)

        v = self.value_head(x).view(B * self.heads, C // self.heads, H, W)
        o = F.grid_sample(v, grid, mode='bilinear',
                          padding_mode='border', align_corners=True)
        return o.view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Block and model
# ─────────────────────────────────────────────────────────────────────────────

class WarpSNOBlock(nn.Module):
    """
    Three parallel paths, summed:

        shearlet spectral mix   —  restores fine-scale directional energy
        shearlet-guided warp    —  moves fronts to the right place
        1x1 bypass              —  pointwise channel mixing

    SNO's block is (spectral + bypass); Flower's argument is that the cheap
    spatial path should be a warp rather than a pointwise conv.  Here it is
    both, because the probe says the two error components are of comparable
    size and neither branch can express the other.
    """

    def __init__(self, channels: int, Psi: torch.Tensor, scale_of: np.ndarray,
                 size: Tuple[int, int], num_heads: int = 8,
                 guided: bool = True, max_disp: float = 3.0,
                 groups: int = 8, energy: str = 'scale'):
        super().__init__()
        self.spectral = ShearletMix(channels, Psi, scale_of, energy=energy)
        g = self.spectral.n_scales_out if guided else 0
        self.guided = guided
        self.warp   = SelfWarp2d(channels, size, num_heads=num_heads,
                                 guide_dim=g, max_disp=max_disp)
        self.bypass = nn.Conv2d(channels, channels, 1)
        self.norm   = nn.GroupNorm(min(groups, channels), channels)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec, eng = self.spectral(x)
        w = self.warp(x, eng if self.guided else None)
        return self.act(self.norm(spec + w + self.bypass(x)))


class WarpSNO(nn.Module):
    """
    SNO with a Flower warp branch in every block.

    Same shell as `sno_urban.SNO`: Lift -> blocks -> Project, so it is a
    drop-in for `shock_common.build_model` and inherits PadWrap unchanged.

    Unlike Flower's U-Net, there is no spatial down/upsampling, so the grid
    need not be divisible by 2^(n_levels-1) — which matters here, because the
    `same` task runs at 129² (161² after PadWrap) and can never be made
    divisible by 4. Multiscale structure comes from the shearlet bank instead.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 hidden_channels: int = 32, n_blocks: int = 4,
                 input_size: Tuple[int, int] = None,
                 n_scales: Optional[int] = None,
                 num_heads: int = 8, guided: bool = True,
                 max_disp: float = 3.0, energy: str = 'scale'):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size=(H, W) is required for WarpSNO.")

        H, W     = input_size
        Psi_np   = build_shearlet_filters(H, W, numOfScales=n_scales)
        n_sh     = Psi_np.shape[2]
        J        = (int(np.floor(0.5 * np.log2(max(H, W))))
                    if n_scales is None else n_scales)
        scale_of = _scale_of_subband(n_sh, J)
        Psi      = torch.from_numpy(Psi_np).float().permute(2, 0, 1)

        C = hidden_channels
        self.lift   = nn.Conv2d(in_channels, C, 1)
        self.blocks = nn.ModuleList([
            WarpSNOBlock(C, Psi, scale_of, (H, W), num_heads=num_heads,
                         guided=guided, max_disp=max_disp, energy=energy)
            for _ in range(n_blocks)])
        self.project = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.GELU(), nn.Conv2d(C, out_channels, 1))

        self.n_shearlets = int(n_sh)
        self.H, self.W   = H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        return self.project(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
