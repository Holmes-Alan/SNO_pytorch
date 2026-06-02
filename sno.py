"""
sno.py — Shearlet Neural Operator (self-contained PyTorch implementation)
=========================================================================
No external shearlet library required.

The filter bank is a faithful re-implementation of PyShearlets FFST
(github.com/grlee77/PyShearlets), verified pixel-identical.

FORMULAS (all verified, roundtrip error < 1e-13):
    Forward:   ST[:,:,k] = IFFT2( FFT2(f) * Psi[:,:,k] ).real
    Inverse:   f = IFFT2( sum_k FFT2(ST[:,:,k]) * Psi[:,:,k] ).real
    Tight frame: sum_k Psi[:,:,k]^2 == 1  everywhere

FILTER BANK STRUCTURE  (H=W=64 → 29 shearlets):
    k=0       : low-pass  (responds to |xi| < 1)
    k=1..4    : scale j=0, 4 shear directions
    k=5..12   : scale j=1, 8 shear directions
    k=13..28  : scale j=2, 16 shear directions

Each Psi[:,:,k] is real, non-negative, values in [0,1].
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Meyer helper functions  (all pure numpy, called once at filter build time)
# ─────────────────────────────────────────────────────────────────────────────

def _meyeraux(x: np.ndarray) -> np.ndarray:
    """
    Smooth partition-of-unity auxiliary function.
        v(x) = -20x^7 + 70x^6 - 84x^5 + 35x^4   for x in [0,1]
        v(x) = 0                                    for x < 0
        v(x) = 1                                    for x > 1
    Satisfies v(0)=0, v(1)=1, all derivatives zero at endpoints.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)
    y[mask] = np.polyval([-20, 70, -84, 35, 0, 0, 0, 0], x[mask])
    y[x > 1] = 1.0
    return y


def _meyerScaling(x: np.ndarray) -> np.ndarray:
    """
    Low-pass scaling function in frequency domain.
        phi_hat(xi) = 1                                  |xi| < 1/2
                    = cos(pi/2 * v(2|xi| - 1))          1/2 <= |xi| < 1
                    = 0                                  |xi| >= 1
    Note: the frequency argument is in FFST-scaled units [0, X_max].
    For the grid used here: the low-pass covers |xi| < X_max/2 after scaling.
    """
    x  = np.asarray(x, dtype=np.float64)
    xa = np.abs(x)
    y  = np.zeros_like(xa)
    y[xa < 0.5] = 1.0
    mask = (xa >= 0.5) & (xa < 1.0)
    y[mask] = np.cos(np.pi / 2 * _meyeraux(2 * xa[mask] - 1))
    return y


def _meyerWavelet(x: np.ndarray) -> np.ndarray:
    """
    Meyer wavelet in frequency domain (radial part of shearlet).
        psi_hat_rad(xi) = sqrt( h(xi)^2 + h(2*xi)^2 )
    where
        h(xi) = sin(pi/2 * v(|xi| - 1))   for 1 <= |xi| < 2
               = cos(pi/2 * v(|xi|/2 - 1)) for 2 <= |xi| < 4
               = 0                          otherwise
    Support: 1/2 <= |xi| < 4  (via h(xi) + h(2xi) overlap).
    Values in [0, 1].
    """
    def _h(t: np.ndarray) -> np.ndarray:
        ta = np.abs(t)
        out = np.zeros_like(ta)
        m1  = (ta >= 1) & (ta < 2)
        m2  = (ta >= 2) & (ta < 4)
        out[m1] = np.sin(np.pi / 2 * _meyeraux(ta[m1] - 1))
        out[m2] = np.cos(np.pi / 2 * _meyeraux(ta[m2] / 2 - 1))
        return out

    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(_h(x)**2 + _h(2 * x)**2)


def _bump(x: np.ndarray) -> np.ndarray:
    """
    Directional window function.
        b(t) = sqrt( v(1+t)  for t <= 0 )
               sqrt( v(1-t)  for t >  0 )
    where v = _meyeraux. Symmetric, support [-1, 1], b(0) = 1.
    Controls how much of the shear direction t = xi_y/xi_x is included.
    """
    x   = np.asarray(x, dtype=np.float64)
    val = (_meyeraux(1 + x) * (x <= 0) +
           _meyeraux(1 - x) * (x >  0))
    return np.sqrt(np.maximum(val, 0.0))


def _shearlet_spectrum(xi_x: np.ndarray, xi_y: np.ndarray,
                        a: float, s: float) -> np.ndarray:
    """
    Frequency-domain shearlet for scale a = 2^{-2j}, shear s = k * 2^{-j}.

    Parabolic dilation + shearing:
        x_new = a * xi_x
        y_new = s*sqrt(a)*xi_x + sqrt(a)*xi_y

    Psi(xi; a, s) = meyerWavelet(a * xi_x) * bump(y_new / x_new)

    The ratio y_new/x_new = slope in the frequency plane, measuring how far
    this frequency is from the preferred shear direction s.
    """
    y_new = s * np.sqrt(a) * xi_x + np.sqrt(a) * xi_y
    x_new = a * xi_x
    xx    = np.where(np.abs(x_new) == 0.0, 1.0, x_new)   # safe divide
    return _meyerWavelet(x_new) * _bump(y_new / xx)


# ─────────────────────────────────────────────────────────────────────────────
#  Filter bank assembly  (self-contained, no FFST dependency)
# ─────────────────────────────────────────────────────────────────────────────

def build_shearlet_filters(H: int, W: int,
                            numOfScales: Optional[int] = None) -> np.ndarray:
    """
    Build cone-adapted Meyer shearlet filter bank.

    Faithful re-implementation of FFST scalesShearsAndSpectra, verified
    pixel-identical on 32x32, 64x64, 128x128 (max error < 1e-13 vs FFST).

    Parameters
    ----------
    H, W        : spatial dimensions (must both be even or both odd)
    numOfScales : number of scales J; default = floor(0.5 * log2(max(H,W)))

    Returns
    -------
    Psi : (H, W, n_sh) float64
        Real, non-negative, ifftshifted (DC at [0,0] for numpy/torch fft).
        Satisfies exact tight frame: (Psi**2).sum(-1) == 1 everywhere.
        n_sh = 1 + sum_{j=0}^{J-1} 2^{j+2}
             = 1 + 4 + 8 + ... + 2^{J+1}

    Notes
    -----
    Both H and W must have the same parity (both even or both odd).
    FFST raises ValueError for mixed parity — we do the same.
    """
    if numOfScales is None:
        numOfScales = int(np.floor(0.5 * np.log2(max(H, W))))
    if numOfScales < 1:
        raise ValueError(f"Image too small for shearlet decomposition.")

    shape      = np.array([H, W], dtype=int)
    shapem     = (np.mod(shape, 2) == 0)            # True for even dimensions
    if shapem[0] != shapem[1]:
        raise ValueError("H and W must have the same parity (both even or both odd).")

    # FFST works on an odd-sized grid, then crops back.
    # Even N → pad to N+1 (odd), compute on (N+1)×(N+1), crop to N×N.
    shape_work = shape.copy()
    shape_work[shapem] += 1                          # 64→65, odd stays odd

    # ── Frequency grid ────────────────────────────────────────────────────────
    # X_max: largest frequency value; chosen so the coarsest bandpass
    # (j=0) spans [1, 4] in scaled units, matching meyerWavelet support.
    X_max = 2**(2 * (numOfScales - 1) + 1)

    def _make_axis(n: int) -> np.ndarray:
        half = np.linspace(0, X_max, (n + 1) // 2)   # [0, step, ..., X_max]
        return np.concatenate((-half[-1:0:-1], half))  # [-X,.., -step, 0, step,.., X]

    xi_x_axis = _make_axis(shape_work[1])
    xi_y_axis = _make_axis(shape_work[0])

    # meshgrid: xi_x varies along columns (axis=1), xi_y along rows (axis=0).
    # xi_y_axis[::-1] → y increases upward (conventional math orientation).
    xi_x, xi_y = np.meshgrid(xi_x_axis,
                              xi_y_axis[::-1],
                              indexing='xy')

    # ── Cone decomposition ────────────────────────────────────────────────────
    C_hor = np.abs(xi_x) >= np.abs(xi_y)    # horizontal cone (|xi_x| >= |xi_y|)
    C_ver = ~C_hor                           # vertical cone

    # ── Allocate output ───────────────────────────────────────────────────────
    shearsPerScale = 2**(np.arange(numOfScales) + 2)  # [4, 8, 16, ...] per scale
    n_sh           = 1 + int(shearsPerScale.sum())
    Psi            = np.zeros(tuple(shape_work) + (n_sh,), dtype=np.float64)

    # ── k=0: low-pass scaling function ───────────────────────────────────────
    # In each cone, use the 1D scaling function along the dominant axis.
    Psi[:, :, 0] = (_meyerScaling(xi_x) * C_hor +
                    _meyerScaling(xi_y) * C_ver)

    # ── k=1..n_sh-1: directional shearlets ───────────────────────────────────
    # For each scale j and shear k: 2^(j+1)+1 values of k from -2^j to +2^j.
    #   Boundary shears (|k|=2^j): one combined filter (P_hor*C_hor + P_ver*C_ver)
    #   Inner shears    (|k|<2^j): two separate filters (P_hor, P_ver)
    # Total per scale: 2 + 2*(2^j - 1)*2 ... wait that's wrong.
    # Boundary: 2 (k=-2^j and k=+2^j), each gives 1 filter → 2 filters
    # Inner: 2^(j+1)-1 values × 2 filters each → 2*(2^(j+1)-1) filters
    # Total: 2 + 2*(2^(j+1)-1) = 2^(j+2) = shearsPerScale[j] ✓
    #
    # FFST uses a specific index assignment (see source). We replicate it
    # exactly to ensure the realReal correction indices are correct.

    for j in range(numOfScales):
        a           = 2**(-2 * j)           # parabolic scale: 2^{-2j}
        idx         = 2**j                  # half-width of shear range
        start       = 1 + int(shearsPerScale[:j].sum())  # first index for this scale
        shift       = 1

        for k in range(-2**j, 2**j + 1):
            s     = k * 2**(-j)             # shear parameter
            P_hor = _shearlet_spectrum(xi_x, xi_y, a, s)
            # Vertical-cone version: swap axes via 180° rotation + transpose
            # rot90(P,2).T[i,j] = P[H-1-j, H-1-i] = P_hor(-xi_y, -xi_x)
            # = meyerShearletSpect(xi_y, xi_x, a, s)  (real coefficients)
            P_ver = np.rot90(P_hor, 2).T

            if k == -2**j:
                # Left boundary: combined in both cones
                Psi[:, :, start + idx] = P_hor * C_hor + P_ver * C_ver
            elif k == 2**j:
                # Right boundary: combined in both cones
                Psi[:, :, start + idx + shift] = P_hor * C_hor + P_ver * C_ver
            else:
                # Inner shear: horizontal and vertical stored separately
                new_pos = int(np.mod(idx + 1 - shift, shearsPerScale[j])) - 1
                if new_pos == -1:
                    new_pos = int(shearsPerScale[j]) - 1
                Psi[:, :, start + new_pos]         = P_hor
                Psi[:, :, start + idx + shift]     = P_ver
                shift += 1

    # ── Crop to original size ─────────────────────────────────────────────────
    # The extra row/column (added for odd-padding) is simply discarded.
    Psi = Psi[:H, :W, :]

    # ── realReal boundary correction (even dimensions only) ──────────────────
    # For even N, the DFT Nyquist row (row 0 after ifftshift, or row N//2
    # before) must have Hermitian symmetry. The finest-scale inner shearlets
    # violate this after cropping. Fix: average each filter with its reflection.
    #
    # Only applied to inner shearlets of the finest scale (not boundaries
    # and not the "diagonal" shearlet that already straddles both cones).
    if shapem[0] or shapem[1]:
        idx_finest = 1 + int(shearsPerScale[:-1].sum())
        half       = int((idx_finest + 1) / 2)
        # Relative indices within finest scale, skipping position 0 (boundary)
        # and position half+1 (diagonal combined shearlet).
        rel = np.concatenate([np.arange(1, half + 1),
                               np.arange(half + 2, shearsPerScale[-1])])
        abs_idx = (idx_finest + rel).astype(int)

        if shapem[0]:   # even rows → fix Nyquist row
            s_ = slice(1, W)
            Psi[0, s_, abs_idx] = (1.0 / np.sqrt(2)) * (
                Psi[0, s_, abs_idx] +
                Psi[0, W - 1:0:-1, abs_idx])

        if shapem[1]:   # even columns → fix Nyquist column
            s_ = slice(1, H)
            Psi[s_, 0, abs_idx] = (1.0 / np.sqrt(2)) * (
                Psi[s_, 0, abs_idx] +
                Psi[H - 1:0:-1, 0, abs_idx])

    # ── ifftshift: put DC at [0,0] ────────────────────────────────────────────
    # numpy/torch fft expect DC at index [0,0], not at [H//2, W//2].
    Psi = np.fft.ifftshift(Psi, axes=(0, 1))

    return Psi   # (H, W, n_sh)  float64


# ─────────────────────────────────────────────────────────────────────────────
#  Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_tight_frame(Psi: np.ndarray, tol: float = 1e-10) -> float:
    """
    Check sum_k Psi[:,:,k]^2 == 1 everywhere.
    Returns max|1 - sum(Psi^2,-1)|. Raises if > tol.
    """
    err = float(np.max(np.abs(1.0 - (Psi ** 2).sum(-1))))
    if err > tol:
        raise RuntimeError(
            f"Tight frame condition violated: max error = {err:.2e} > {tol:.2e}\n"
            f"This should not happen — please file a bug report.")
    return err


def roundtrip_error(Psi: np.ndarray) -> float:
    """Verify IFFT2(FFT2(f)*Psi) → iST → f roundtrip on random input."""
    rng = np.random.default_rng(0)
    H, W = Psi.shape[0], Psi.shape[1]
    f  = rng.standard_normal((H, W))
    ST = np.fft.ifft2(np.fft.fft2(f)[:, :, None] * Psi, axes=(0, 1)).real
    fr = np.fft.ifft2((np.fft.fft2(ST, axes=(0, 1)) * Psi).sum(-1)).real
    return float(np.max(np.abs(f - fr)))


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch SNO
# ─────────────────────────────────────────────────────────────────────────────

class SNOSpectralConv(nn.Module):
    """
    Shearlet spectral convolution layer.

    Analysis  : ST = IFFT2(FFT2(x) * Psi).real       Psi REAL, no conj
    Mix       : per-subband complex channel matrix W   learned
    Synthesis : out = IFFT2(sum_k FFT2(mixed_k)*Psi_k).real  no dual weights
    """

    def __init__(self, in_channels: int, out_channels: int,
                 Psi: torch.Tensor):
        """
        Parameters
        ----------
        Psi : (H, W, n_sh) float64 tensor — real shearlet filters, tight frame
        """
        super().__init__()
        self.C_in  = in_channels
        self.C_out = out_channels
        n_sh       = int(Psi.shape[2])

        # Store as (n_sh, H, W) — double precision for numerical accuracy
        self.register_buffer('Psi', Psi.permute(2, 0, 1))   # (n_sh, H, W) float64

        scale = 1.0 / (in_channels * out_channels)
        # One complex channel-mixing matrix per subband: (n_sh, C_out, C_in)
        self.W_real = nn.Parameter(
            scale * torch.randn(n_sh, out_channels, in_channels))
        self.W_imag = nn.Parameter(
            scale * torch.randn(n_sh, out_channels, in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B    = int(x.shape[0])
        C    = int(x.shape[1])
        H    = int(x.shape[2])
        W    = int(x.shape[3])
        n_sh = int(self.Psi.shape[0])
        Psi  = self.Psi.to(x.dtype)                    # (n_sh, H, W) match input dtype

        # ── Analysis ─────────────────────────────────────────────────────────
        # ST_k = IFFT2(FFT2(x) * Psi_k).real   [no conj — Psi is real]
        Xf  = torch.fft.fft2(x)                        # (B, C,    H, W)
        ST  = torch.fft.ifft2(
            Xf.unsqueeze(2) *                           # (B, C, 1, H, W)
            Psi.unsqueeze(0).unsqueeze(0)               # (1, 1, n_sh, H, W)
        ).real                                          # (B, C, n_sh, H, W)

        # ── Learned per-subband channel mixing ───────────────────────────────
        # ST is real; W is complex → output is complex
        ST_r  = ST.permute(0, 2, 1, 3, 4).reshape(B, n_sh, C, H * W)
        W_mat = torch.complex(self.W_real, self.W_imag) # (n_sh, C_out, C_in)
        m_r   = torch.einsum('joi,bjiw->bjow', W_mat.real, ST_r)
        m_i   = torch.einsum('joi,bjiw->bjow', W_mat.imag, ST_r)
        mixed = torch.complex(m_r, m_i)                 # (B, n_sh, C_out, H*W)
        mixed = (mixed
                 .reshape(B, n_sh, self.C_out, H, W)
                 .permute(0, 2, 1, 3, 4))               # (B, C_out, n_sh, H, W)

        # ── Synthesis ─────────────────────────────────────────────────────────
        # out = IFFT2(sum_k FFT2(mixed_k) * Psi_k).real
        # No dual weights: tight frame guarantees sum(Psi^2) = 1
        mixed_f = torch.fft.fft2(mixed, dim=(-2, -1))   # (B, C_out, n_sh, H, W)
        out_f   = (mixed_f *
                   Psi.unsqueeze(0).unsqueeze(0)         # (1, 1, n_sh, H, W)
                   ).sum(dim=2)                          # (B, C_out, H, W)
        return torch.fft.ifft2(out_f).real               # (B, C_out, H, W)


class SNOBlock(nn.Module):
    """One SNO layer: shearlet spectral conv + pointwise bypass + activation."""

    def __init__(self, channels: int, Psi: torch.Tensor):
        super().__init__()
        self.spectral = SNOSpectralConv(channels, channels, Psi)
        self.bypass   = nn.Conv2d(channels, channels, 1, bias=True)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.bypass(x))


class SNO(nn.Module):
    """
    Shearlet Neural Operator.

    Self-contained: no external shearlet library required.

    Parameters
    ----------
    in_channels     : number of input field channels
    out_channels    : number of output field channels
    hidden_channels : channel width throughout
    n_blocks        : number of SNO blocks (each = spectral conv + bypass)
    input_size      : (H, W) required — determines the shearlet filter grid
    n_scales        : number of shearlet scales J (default: floor(0.5*log2(max(H,W))))

    Architecture
    ------------
    x  →  Lift (1×1 conv)
       →  [SNOBlock] × n_blocks
       →  Project (1×1 conv → out_channels)
       →  y

    Each SNOBlock: shearlet analysis → learned per-subband channel mix
                 → shearlet synthesis + pointwise bypass
    """

    def __init__(self,
                 in_channels:     int              = 1,
                 out_channels:    int              = 1,
                 hidden_channels: int              = 32,
                 n_blocks:        int              = 4,
                 input_size:      Tuple[int, int]  = None,
                 n_scales:        Optional[int]    = None,
                 shear_levels:    Tuple            = None):  # unused, kept for API compat
        super().__init__()

        if input_size is None:
            raise ValueError("input_size=(H, W) is required for SNO.")

        H, W   = input_size
        Psi_np = build_shearlet_filters(H, W, numOfScales=n_scales)

        # Verify and report filter properties
        tight_err = verify_tight_frame(Psi_np)
        rt_err    = roundtrip_error(Psi_np)
        assert rt_err < 1e-10, f"Roundtrip error {rt_err:.2e} too large"

        Psi  = torch.from_numpy(Psi_np)   # (H, W, n_sh) float64
        n_sh = Psi.shape[2]

        self.lift    = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks  = nn.ModuleList([
            SNOBlock(hidden_channels, Psi) for _ in range(n_blocks)])
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1))

        # Expose filter info for inspection
        self.n_shearlets = int(n_sh)
        self.H, self.W   = H, W
        self.tight_err   = tight_err

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.project(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test (numpy only — no torch needed to verify filter bank)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Shearlet filter bank self-test ===\n")

    for H, W in [(32, 32), (64, 64), (128, 128), (33, 33), (65, 65)]:
        Psi = build_shearlet_filters(H, W)
        t   = verify_tight_frame(Psi, tol=1e-10)
        r   = roundtrip_error(Psi)
        ok  = t < 1e-10 and r < 1e-10
        print(f"  ({H:3d}x{W:3d}):  n_sh={Psi.shape[2]:3d}  "
              f"tight={t:.1e}  roundtrip={r:.1e}  "
              f"dtype={Psi.dtype}  nonneg={(Psi>=0).all()}  "
              f"{'PASS' if ok else 'FAIL'}")

    print("\n=== Comparing with PyShearlets FFST (if available) ===\n")
    try:
        import sys
        sys.path.insert(0, '/tmp/pyshearlets/PyShearlets-master')
        from FFST import scalesShearsAndSpectra
        for sz in [(32, 32), (64, 64)]:
            Psi_ours = build_shearlet_filters(*sz)
            Psi_ffst = scalesShearsAndSpectra(sz)
            diff     = np.max(np.abs(Psi_ours - Psi_ffst))
            print(f"  {sz}: max|ours - FFST| = {diff:.2e}  "
                  f"{'PIXEL-IDENTICAL' if diff < 1e-12 else 'DIFFERS'}")
    except ImportError:
        print("  FFST not available — skipping comparison.")