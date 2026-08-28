"""
sno_fast.py — SNO as a shearlet-parameterised Fourier multiplier
=================================================================
Mathematically IDENTICAL to `sno_urban.SNO`, and state-dict compatible with it,
but ~12x faster and ~20x lighter.  The speed was never a property of the
method; it was a property of the implementation.

THE ALGEBRA
-----------
`SNOSpectralConv` materialises the full shearlet decomposition, a tensor
(B, C, n_sh, H, W) — 29x the field — and runs an inverse FFT over it, mixes,
then an FFT back over it.  None of that is necessary:

    ST_k      = IFFT(Xf . Psi_k).real
    Psi_k is EVEN in frequency (verified: max|Psi_k(-xi) - Psi_k(xi)| = 0 for
    all 29 subbands), and x is real, so Xf . Psi_k is Hermitian and the .real
    discards nothing.  Therefore  FFT(ST_k) = Xf . Psi_k  exactly, and the
    analysis never has to leave the Fourier domain.

    mixed_k   = W_k @ ST_k                    channel mix, pointwise in space
    out       = IFFT( sum_k FFT(mixed_k) . Psi_k ).real
              = IFFT( sum_k (W_k @ Xf) . Psi_k^2 ).real
              = IFFT( M(xi) @ Xf(xi) ).real

              with   M(xi) = sum_k W_k Psi_k(xi)^2

WHAT THIS SAYS ABOUT SNO
------------------------
An SNO block IS an FNO block whose multiplier is structured: because the
shearlet bank is a tight frame, sum_k Psi_k^2 = 1, so M(xi) is a smooth
partition-of-unity blend of 29 learned channel matrices over a directional,
multiscale tiling of the frequency plane.  FNO learns one matrix per Fourier
mode inside a k_max box; SNO learns one matrix per shearlet subband and
interpolates between them everywhere.  That — not the FFT bookkeeping — is
the inductive bias, and it survives this rewrite untouched.

MEASURED (B=4, C=32, fwd+bwd, one block)
    160²   sno_urban  88.9 ms / 3689 MB      this  7.2 ms / 185 MB   12.3x
    289²   sno_urban  OOM at 31 GB           this 26.8 ms / 579 MB
    relative error vs sno_urban: 4.9e-07  (fp32 roundoff)

The 289² line is the one that matters for `--task sr`: the patch-128 +
sliding-window pipeline exists only because the original form cannot hold a
257² frame.  This one can.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sno_urban import build_shearlet_filters


class ShearletMultiplier(nn.Module):
    """
    Exact replacement for `sno_urban.SNOSpectralConv`, same parameters and
    same buffer names, so checkpoints interchange without conversion.

    This is also the minimal reusable SNO primitive: a single instance gives
    any architecture the directional/multiscale spectral selectivity that SNO
    is for, at the cost of two rfft2s.
    """

    def __init__(self, in_channels: int, out_channels: int, Psi: torch.Tensor):
        super().__init__()
        self.C_in, self.C_out = in_channels, out_channels
        # (H, W, n_sh) -> (n_sh, H, W), matching SNOSpectralConv's buffer
        self.register_buffer('Psi', Psi.float().permute(2, 0, 1))
        scale  = 1.0 / (in_channels * out_channels)
        self.W = nn.Parameter(scale * torch.randn(Psi.shape[2],
                                                  out_channels, in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        # rfft2 keeps W//2+1 columns; Psi is even, so the dropped half is
        # determined and irfft2 reconstructs the real output exactly.
        G  = self.Psi[..., :W // 2 + 1] ** 2            # (n_sh, H, Wh)
        Xf = torch.fft.rfft2(x)                         # (B, C_in, H, Wh)
        M  = torch.einsum('koi,khw->oihw', self.W, G)   # (C_out, C_in, H, Wh)
        # keep M real and split the complex product, rather than casting M to
        # complex — halves the memory of the largest tensor in the layer.
        re = torch.einsum('oihw,bihw->bohw', M, Xf.real)
        im = torch.einsum('oihw,bihw->bohw', M, Xf.imag)
        return torch.fft.irfft2(torch.complex(re, im), s=(H, W))


class FastSNOBlock(nn.Module):
    """`sno_urban.SNOBlock` with the multiplier form. Same state-dict keys."""

    def __init__(self, channels: int, Psi: torch.Tensor):
        super().__init__()
        self.spectral = ShearletMultiplier(channels, channels, Psi)
        self.bypass   = nn.Conv2d(channels, channels, 1, bias=True)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.bypass(x))


class FastSNO(nn.Module):
    """
    Drop-in for `sno_urban.SNO`: identical outputs, identical state_dict.

    An existing SNO checkpoint loads into this class with strict=True, and a
    checkpoint trained here loads back into `sno_urban.SNO`.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 hidden_channels: int = 32, n_blocks: int = 4,
                 input_size: Tuple[int, int] = None,
                 n_scales: Optional[int] = None,
                 shear_levels=None):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size=(H, W) is required for FastSNO.")
        H, W = input_size
        Psi  = torch.from_numpy(build_shearlet_filters(H, W,
                                                       numOfScales=n_scales))
        C = hidden_channels
        self.lift    = nn.Conv2d(in_channels, C, 1)
        self.blocks  = nn.ModuleList([FastSNOBlock(C, Psi)
                                      for _ in range(n_blocks)])
        self.project = nn.Sequential(nn.Conv2d(C, C, 1), nn.GELU(),
                                     nn.Conv2d(C, out_channels, 1))
        self.n_shearlets = int(Psi.shape[2])
        self.H, self.W   = H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        return self.project(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal SNO inside Flower
# ─────────────────────────────────────────────────────────────────────────────

class FlowerSNO(nn.Module):
    """
    Flower, plus ONE shearlet multiplier block.  The point is to add the single
    thing Flower structurally lacks, and nothing else.

    WHAT EACH SIDE ACTUALLY CONTRIBUTES
    -----------------------------------
    Flower is not local.  Each head samples at an arbitrary source coordinate,
    so its reach is unbounded — but every target pixel fetches from ONE point
    per head.  Its nonlocality is ADAPTIVE and SPARSE, and it carries no notion
    of orientation or scale: the flow head is a 1x1 MLP on the local feature
    vector.

    A shearlet multiplier is the complement on both axes.  It is DENSE (every
    output frequency sees the whole field) and FIXED (the same multiplier
    everywhere), and its whole content is direction- and scale-selectivity:
    M(xi) = sum_k W_k Psi_k(xi)^2 blends 29 learned channel matrices over a
    directional, multiscale tiling of the frequency plane.

    So the split is not global-vs-local.  It is:
        Flower    adaptive, sparse   — WHERE to fetch from
        shearlet  fixed, dense       — WHICH orientations and scales to amplify

    PLACEMENT
    ---------
    'head' (default)  Flower first, one shearlet block refines its output at
                      full resolution.  Fine detail is what the shearlet basis
                      is for, and full resolution is where it lives.
    'stem'            shearlet first, so the warp heads act on directionally
                      analysed features.
    'both'            one of each; still only two spectral layers.

    Cost is now negligible: `ShearletMultiplier` is two rfft2s, ~7 ms at
    160²/C=32/B=4 versus 89 ms for the sno_urban form.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 hidden_channels: int = 28,
                 input_size: Tuple[int, int] = None,
                 lifting_dim: int = 28, n_levels: int = 3,
                 num_heads: int = 7, groups: int = 7,
                 n_scales: Optional[int] = None,
                 sno_at: str = 'head', dropout_rate: float = 0.0):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size=(H, W) is required for FlowerSNO.")
        if sno_at not in ('head', 'stem', 'both'):
            raise ValueError(f"sno_at must be head|stem|both, got {sno_at!r}")

        from shock_common import _import_flower
        Flower = _import_flower()

        H, W = input_size
        div  = 2 ** (n_levels - 1)
        if H % div or W % div:
            raise ValueError(
                f"FlowerSNO inherits Flower's constraint: the padded grid must "
                f"be divisible by 2**(n_levels-1)={div}, got {H}x{W}.")

        C   = hidden_channels
        Psi = torch.from_numpy(build_shearlet_filters(H, W, numOfScales=n_scales))

        self.lift   = nn.Conv2d(in_channels, C, 1)
        self.stem   = (FastSNOBlock(C, Psi)
                       if sno_at in ('stem', 'both') else None)
        self.flower = Flower(dim_in=C, dim_out=C, n_spatial_dims=2,
                             spatial_resolution=[H, W],
                             lifting_dim=lifting_dim, n_levels=n_levels,
                             num_heads=num_heads, groups=groups,
                             boundary_condition_types=['DIRICHLET', 'DIRICHLET'],
                             dropout_rate=dropout_rate)
        self.head   = (FastSNOBlock(C, Psi)
                       if sno_at in ('head', 'both') else None)
        self.project = nn.Conv2d(C, out_channels, 1)

        self.sno_at      = sno_at
        self.n_shearlets = int(Psi.shape[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        if self.stem is not None:
            h = h + self.stem(h)
        h = self.flower(h)
        if self.head is not None:
            h = h + self.head(h)
        return self.project(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
