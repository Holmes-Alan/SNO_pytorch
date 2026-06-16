"""
fno.py — Fourier Neural Operator (matched-parameter version)
============================================================
Two bugs fixed vs the original implementation:

  BUG 1 — Wrong number of spectral quadrants
    Old: 4 quadrant weights  (w_real shape [4, C, C, k, k])
    Fix: 2 quadrant weights  — rfft2 produces (H, W//2+1) output;
         only top-left and bottom-left k×k blocks are non-redundant.
         4 quadrants was storing 2× more parameters than needed.

  BUG 2 — k_max not matched to SNO parameter count
    Old: k_max fixed at 12 or 16 → FNO had 2-5× more params than SNO
    Fix: k_max derived from n_sh (number of shearlets) so spectral
         parameter counts match:
            FNO spectral/block = 2 × 2 × k² × C²   (2 quad, real+imag)
            SNO spectral/block = 2 × n_sh × C²       (n_sh subbands)
            Match: k_max = round( sqrt(n_sh / 2) )
         For 64×64: n_sh=29 → k_max=4, giving 1.10× SNO params.

PARAMETER COMPARISON (hidden=32, n_blocks=4, H=W=64):
    FNO (corrected): 267,521  params  (k_max=4)
    SNO:             242,945  params  (n_sh=29)
    Ratio: 1.10×  — essentially matched

FORMULAS:
    Forward:   x̂ = rfft2(x)
               out_k = R_k ⊗ x̂_k   (k ∈ {top-left, bottom-left} k×k block)
               y = irfft2(out)  +  bypass(x)
"""

from __future__ import annotations
from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter-matching utility
# ─────────────────────────────────────────────────────────────────────────────

def matched_k_max(n_sh: int) -> int:
    """
    Compute the FNO k_max that matches SNO spectral parameter count.

    SNO spectral params/block = 2 * n_sh * C²   (W_real + W_imag)
    FNO spectral params/block = 2 * 2 * k² * C² (2 quadrants, real + imag)
    Match: 4k² = 2*n_sh  →  k = sqrt(n_sh/2)

    Parameters
    ----------
    n_sh : number of shearlets (from build_shearlet_filters)

    Returns
    -------
    k_max : int, minimum 1
    """
    return max(1, round(math.sqrt(n_sh / 2)))


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral convolution — 2 quadrant rfft2, correct and efficient
# ─────────────────────────────────────────────────────────────────────────────

class FNOSpectralConv2d(nn.Module):
    """
    2D FNO spectral convolution using rfft2 with 2 quadrant weights.

    rfft2(x) has shape (H, W//2+1) — only non-negative W-frequencies.
    We keep k_max modes in both H and W directions:
      - Top-left block    : rows  0..k_max-1,   cols 0..k_max-1
      - Bottom-left block : rows  H-k_max..H-1, cols 0..k_max-1

    Learned weights: R ∈ ℂ^{2 × C_in × C_out × k_max × k_max}
    (2 quadrants, complex = stored as real + imag pair)
    """

    def __init__(self, in_channels: int, out_channels: int, k_max: int):
        super().__init__()
        self.C_in  = in_channels
        self.C_out = out_channels
        self.k_max = k_max

        scale = 1.0 / (in_channels * out_channels)
        # 2 quadrants × (C_in → C_out) × (k_max × k_max), stored as real+imag
        self.w_real = nn.Parameter(
            scale * torch.randn(2, in_channels, out_channels, k_max, k_max))
        self.w_imag = nn.Parameter(
            scale * torch.randn(2, in_channels, out_channels, k_max, k_max))

    @property
    def weights(self) -> torch.Tensor:
        return torch.complex(self.w_real, self.w_imag)  # (2, C_in, C_out, k, k)

    def _mul(self, x_f: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        x_f : (B, C_in, k, k) complex
        W   : (C_in, C_out, k, k) complex
        →     (B, C_out, k, k) complex
        """
        return torch.einsum('bixy,ioXY->boXY', x_f, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        C = int(x.shape[1])
        H = int(x.shape[2])
        W = int(x.shape[3])
        k = self.k_max

        x_ft = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1)

        out_ft = torch.zeros(B, self.C_out, H, W // 2 + 1,
                              dtype=torch.cfloat, device=x.device)

        Ws = self.weights  # (2, C_in, C_out, k, k)

        # Top-left quadrant: rows 0..k-1, cols 0..k-1
        out_ft[:, :, :k, :k] = self._mul(x_ft[:, :, :k, :k], Ws[0])

        # Bottom-left quadrant: rows H-k..H-1, cols 0..k-1
        out_ft[:, :, -k:, :k] = self._mul(x_ft[:, :, -k:, :k], Ws[1])

        return torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')


# ─────────────────────────────────────────────────────────────────────────────
#  FNO block and full model
# ─────────────────────────────────────────────────────────────────────────────

class FNOBlock(nn.Module):
    """One FNO block: spectral conv + pointwise bypass + activation."""

    def __init__(self, channels: int, k_max: int):
        super().__init__()
        self.spectral = FNOSpectralConv2d(channels, channels, k_max)
        self.bypass   = nn.Conv2d(channels, channels, 1, bias=True)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.bypass(x))


class FNO(nn.Module):
    """
    Fourier Neural Operator — parameter-matched to SNO.

    Parameters
    ----------
    in_channels     : input field channels
    out_channels    : output field channels
    hidden_channels : channel width  (C)
    k_max           : Fourier mode cutoff per spatial direction.
                      If None and n_sh is given, computed automatically as
                      round(sqrt(n_sh/2)) to match SNO parameter count.
    n_sh            : number of shearlets in the paired SNO model (optional).
                      Used only for automatic k_max matching.
    n_blocks        : number of FNO blocks
    input_size      : not required for FNO (resolution-independent modes)
    """

    def __init__(self, in_channels:     int              = 1,
                       out_channels:    int              = 1,
                       hidden_channels: int              = 32,
                       k_max:           Optional[int]    = None,
                       n_sh:            Optional[int]    = None,
                       n_blocks:        int              = 4,
                       input_size:      Tuple[int,int]   = None,   # unused, kept for API compat
                       shear_levels:    Tuple            = None,   # unused
                       n_scales:        Optional[int]    = None):  # unused
        super().__init__()

        # Resolve k_max
        if k_max is None:
            if n_sh is not None:
                k_max = matched_k_max(n_sh)
            else:
                raise ValueError(
                    "Provide either k_max or n_sh (to auto-match SNO params).")

        self.k_max = k_max

        self.lift    = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks  = nn.ModuleList([
            FNOBlock(hidden_channels, k_max) for _ in range(n_blocks)])
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1), nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.project(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== FNO parameter matching ===\n")
    C, n_blocks = 32, 4
    for H in [32, 64, 128]:
        numOfScales = int(np.floor(0.5 * np.log2(H)))
        shearsPerScale = 2**(np.arange(numOfScales)+2)
        n_sh = 1 + int(shearsPerScale.sum())
        k    = matched_k_max(n_sh)

        # SNO spectral params per block
        sno_spectral = 2 * n_sh * C * C
        # FNO spectral params per block (2 quadrants, real+imag)
        fno_spectral = 2 * 2 * k * k * C * C
        ratio = fno_spectral / sno_spectral

        print(f"H=W={H}: n_sh={n_sh}  k_max={k}")
        print(f"  SNO spectral/block : {sno_spectral:>8,}")
        print(f"  FNO spectral/block : {fno_spectral:>8,}  ({ratio:.2f}× SNO)")

    print("\n=== Forward pass test ===\n")
    for n_sh, name in [(13,'32x32'), (29,'64x64')]:
        k = matched_k_max(n_sh)
        model = FNO(n_sh=n_sh, hidden_channels=C, n_blocks=n_blocks)
        x = torch.randn(2, 1, 64, 64)
        y = model(x)
        print(f"  {name}: k_max={k}  params={model.param_count():,}  "
              f"output={y.shape}  OK")