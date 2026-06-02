"""
usno.py — U-Net Shearlet Neural Operator (Multi-scale SNO)
==========================================================

WHY PREVIOUS USNO FAILED
-------------------------
Issue 1 — Single FFT pair, no iterative spatial refinement:
    USNO: x → FFT2 → [mixer×5] → IFFT2 → y
    All 5 mixers see the SAME shearlet decomposition of x.
    No new spatial features are extracted between layers.
    SNO works because each block returns to spatial domain and the
    next block sees the corrected field — iterative refinement.

Issue 2 — No cross-scale correlations:
    SubbandMixer W: (n_act, C, C) mixes channels WITHIN each subband.
    Subband j=0 cannot influence subband j=2 within any single layer.
    SNO's spatial bypass Conv2d(C,C,1) acts on the synthesised spatial
    field, which contains all scales mixed — cross-scale correlations
    emerge naturally via the spatial domain.

SOLUTION: Multi-scale SNO
-------------------------
Stack SNO-style blocks (each with its own FFT2/IFFT2 pair and spatial
bypass) but apply a U-Net scale schedule to each block's shearlet mask.

  Block 0 [encoder, all scales ]:  x    → shearlet(all)  → spatial → skip_0
  Block 1 [encoder, j=0,1 only]:  x_1  → shearlet(j≤2)  → spatial → skip_1
  Block 2 [bottleneck, j=0 only]: x_2  → shearlet(j≤1)  → spatial
  Block 3 [decoder,   j=0,1    ]:  x_3  + skip_1  → shearlet(j≤2)  → spatial
  Block 4 [decoder,   all scales]: x_4  + skip_0  → shearlet(all)  → spatial

Benefits:
  ✓ Multiple FFT pairs → iterative spatial refinement (depth, like SNO)
  ✓ Progressive scale masking → U-Net coarse-to-fine structure
  ✓ Spatial bypass in every block → cross-scale correlations
  ✓ Skip connections in spatial domain → short gradient paths
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from sno import build_shearlet_filters


# ─────────────────────────────────────────────────────────────────────────────
#  Scale structure
# ─────────────────────────────────────────────────────────────────────────────

def _scale_of_subband(n_sh: int, numOfScales: int) -> np.ndarray:
    shearsPerScale = 2**(np.arange(numOfScales) + 2)
    out = np.zeros(n_sh, dtype=int)
    idx = 1
    for j, ns in enumerate(shearsPerScale):
        out[idx:idx + int(ns)] = j + 1
        idx += int(ns)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Masked shearlet spectral convolution — one SNO-style block
# ─────────────────────────────────────────────────────────────────────────────

class MaskedShearletConv(nn.Module):
    """
    One SNO-style spectral convolution with a scale mask.

    Only subbands in active_mask participate in analysis and synthesis.
    Inactive subbands are ignored — their Psi columns are simply not used.
    Spatial bypass ensures cross-scale information is always preserved.

    Forward:
        ST   = IFFT2(FFT2(x) * Psi_active).real      partial shearlet analysis
        mix  = W ⊗ ST                                 per-subband channel mix (real)
        y    = IFFT2(sum_active FFT2(mix_k)*Psi_k)   partial synthesis
        out  = GELU(y + bypass(x))                    back to spatial domain
    """

    def __init__(self, channels: int, Psi: torch.Tensor,
                 active_mask: List[bool]):
        """
        Parameters
        ----------
        Psi         : (n_sh, H, W) float64 — full filter bank
        active_mask : bool list length n_sh — which subbands this block uses
        """
        super().__init__()
        self.C = channels

        act_idx  = [i for i, a in enumerate(active_mask) if a]
        n_act    = len(act_idx)
        self.register_buffer('act_idx',
                             torch.tensor(act_idx, dtype=torch.long))
        # Keep only active filters: (n_act, H, W)
        self.register_buffer('Psi_act', Psi[act_idx].clone())

        # Real per-subband channel mixing (no complex ops needed)
        scale    = 1.0 / channels
        self.W   = nn.Parameter(scale * torch.randn(n_act, channels, channels))

        # Spatial bypass — enables cross-scale correlations
        self.bypass = nn.Conv2d(channels, channels, 1, bias=True)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        C = int(x.shape[1])
        H = int(x.shape[2])
        W = int(x.shape[3])
        n_act = int(self.Psi_act.shape[0])

        Psi = self.Psi_act.to(x.dtype)                    # (n_act, H, W)

        # ── Partial analysis ─────────────────────────────────────────────────
        Xf = torch.fft.fft2(x)                            # (B, C, H, W) cfloat
        ST = torch.fft.ifft2(
            Xf.unsqueeze(2) * Psi.unsqueeze(0).unsqueeze(0)
        ).real                                             # (B, C, n_act, H, W) real

        # ── Per-subband channel mixing (real) ─────────────────────────────────
        ST_r  = ST.permute(0, 2, 1, 3, 4).reshape(B, n_act, C, H * W)
        mixed = torch.einsum('joi,bjiw->bjow', self.W, ST_r)  # (B, n_act, C, H*W)
        mixed = (mixed
                 .reshape(B, n_act, C, H, W)
                 .permute(0, 2, 1, 3, 4))                 # (B, C, n_act, H, W)

        # ── Partial synthesis ─────────────────────────────────────────────────
        mixed_f = torch.fft.fft2(mixed, dim=(-2, -1))     # (B, C, n_act, H, W)
        y_f     = (mixed_f * Psi.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        y       = torch.fft.ifft2(y_f).real               # (B, C, H, W)

        # ── Spatial bypass + activation ───────────────────────────────────────
        return self.act(y + self.bypass(x))


# ─────────────────────────────────────────────────────────────────────────────
#  USNO — Multi-scale SNO with U-Net structure
# ─────────────────────────────────────────────────────────────────────────────

class USNO(nn.Module):
    """
    U-Net Shearlet Neural Operator.

    Stacked SNO-style blocks with progressive scale masking and spatial
    skip connections. Each block has its own FFT2/IFFT2 pair, spatial
    bypass, and returns to spatial domain — enabling iterative refinement
    and cross-scale correlations.
    """

    def __init__(self, in_channels:     int             = 1,
                       out_channels:    int             = 1,
                       hidden_channels: int             = 32,
                       n_scales:        Optional[int]   = None,
                       shear_levels:    Tuple           = None,   # unused
                       n_layers:        int             = 5,
                       input_size:      Tuple[int, int] = None):
        super().__init__()

        if input_size is None:
            raise ValueError("input_size=(H, W) is required for USNO.")

        H, W        = input_size
        Psi_np      = build_shearlet_filters(H, W, numOfScales=n_scales)
        n_sh        = Psi_np.shape[2]
        numOfScales = (int(np.floor(0.5 * np.log2(max(H, W))))
                       if n_scales is None else n_scales)
        scale_of    = _scale_of_subband(n_sh, numOfScales)

        # (n_sh, H, W) float64 — passed into each block's constructor
        Psi = torch.from_numpy(Psi_np).permute(2, 0, 1)

        C = hidden_channels
        self.lift    = nn.Conv2d(in_channels, C, 1)
        self.project = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.GELU(),
            nn.Conv2d(C, out_channels, 1))

        # U-Net scale schedule
        half  = n_layers // 2
        j_bot = max(1, numOfScales - half)

        def mask(j_max: int) -> List[bool]:
            return [bool(scale_of[k] <= j_max) for k in range(n_sh)]

        enc_js = [min(numOfScales, numOfScales - l) for l in range(half)]
        dec_js = [min(numOfScales, j_bot + (l + 1))
                  for l in range(n_layers - half - 1)]

        self.enc_blocks = nn.ModuleList([
            MaskedShearletConv(C, Psi, mask(j)) for j in enc_js])
        self.bot_block  = MaskedShearletConv(C, Psi, mask(j_bot))
        self.dec_blocks = nn.ModuleList([
            MaskedShearletConv(C, Psi, mask(j)) for j in dec_js])

        # Skip connection merges: cat([x_dec, skip_enc]) → C channels
        n_skip = min(len(self.dec_blocks), half)
        self.skip_merge = nn.ModuleList([
            nn.Conv2d(C * 2, C, 1) for _ in range(n_skip)])

        self.n_sh = n_sh
        self.H, self.W = H, W
        self._n_skip = n_skip

        # Report schedule
        print(f"  USNO schedule: enc_j={enc_js}  bot_j={j_bot}  dec_j={dec_js}")
        print(f"  Active subbands: enc={[sum(mask(j)) for j in enc_js]}"
              f"  bot={sum(mask(j_bot))}  dec={[sum(mask(j)) for j in dec_js]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)                                   # (B, C, H, W)

        # ── Encoder: apply blocks, return spatial output, store skip ─────────
        skips = []
        for blk in self.enc_blocks:
            x = blk(x)                                     # (B, C, H, W) spatial
            skips.append(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.bot_block(x)

        # ── Decoder: merge spatial skip, then apply block ─────────────────────
        for i, blk in enumerate(self.dec_blocks):
            skip_idx = len(skips) - 1 - i
            if 0 <= skip_idx < len(skips) and i < self._n_skip:
                # Concatenate in spatial domain, project back to C channels
                x = self.skip_merge[i](
                    torch.cat([x, skips[skip_idx]], dim=1))  # (B, 2C→C, H, W)
            x = blk(x)

        return self.project(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)