"""
cascade_usno.py — Cascade USNO
==============================
Same training as USNO (end-to-end). Architecture differs in three ways:

  1. Decoder blocks are UNMASKED (full shearlet basis).
     Encoder masks guide coarse-to-fine extraction.
     Decoder should learn corrections at all scales.

  2. Global shortcut: x_lifted → Conv → global_h
     Parallel path that bypasses the entire U-Net.
     Acts as a learnable baseline prediction.

  3. Encoder skips store RESIDUALS to the global shortcut:
        skip_i = enc_output_i − global_h
     Decoder adds back: h = skip_i + dec_block(h)
     → decoder learns what each encoder level captured
       BEYOND the global baseline, not the full signal.

Forward:
    x_lifted = Lift(x)
    global_h = GlobalShortcut(x_lifted)       ← parallel baseline

    h = x_lifted
    for enc_blk:                               ← masked encoder
        h = enc_blk(h)
        skips.append(h − global_h)            ← residual skip

    h = bot_block(h)                          ← coarsest bottleneck

    for dec_blk:                              ← UNMASKED decoder
        h = skips[i] + dec_blk(h)            ← residual reconstruction

    return Project(GELU(h + global_h))        ← add baseline back

Training: end-to-end (same as USNO, no cascade phases).
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from sno  import build_shearlet_filters
from usno import MaskedShearletConv, _scale_of_subband


# ─────────────────────────────────────────────────────────────────────────────
#  Kept for API compatibility with train_all.py
# ─────────────────────────────────────────────────────────────────────────────

def mlmc_band_budget(K: int, d: int = 2, alpha: float = 2.0,
                      total_budget: int = 200) -> List[int]:
    """Unused — CascadeUSNO trains end-to-end. Kept for import compatibility."""
    return [total_budget]


# ─────────────────────────────────────────────────────────────────────────────
#  CascadeUSNO
# ─────────────────────────────────────────────────────────────────────────────

class CascadeUSNO(nn.Module):

    def __init__(self, in_channels:     int             = 1,
                       out_channels:    int             = 1,
                       hidden_channels: int             = 32,
                       n_scales:        Optional[int]   = None,
                       shear_levels:    Tuple           = None,
                       n_layers:        int             = 5,
                       input_size:      Tuple[int, int] = None):
        super().__init__()

        if input_size is None:
            raise ValueError("input_size=(H, W) is required.")

        H, W        = input_size
        Psi_np      = build_shearlet_filters(H, W, numOfScales=n_scales)
        n_sh        = Psi_np.shape[2]
        numOfScales = (int(np.floor(0.5 * np.log2(max(H, W))))
                       if n_scales is None else n_scales)
        scale_of    = _scale_of_subband(n_sh, numOfScales)

        Psi = torch.from_numpy(Psi_np).permute(2, 0, 1)  # (n_sh, H, W)

        C = hidden_channels
        self.lift    = nn.Conv2d(in_channels, C, 1)
        self.project = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.GELU(),
            nn.Conv2d(C, out_channels, 1))
        self.act     = nn.GELU()

        # Global shortcut: learnable baseline from x_lifted
        self.global_shortcut = nn.Conv2d(C, C, 1, bias=True)

        half  = n_layers // 2
        j_bot = max(1, numOfScales - half)

        def mask(j_max):
            return [bool(scale_of[k] <= j_max) for k in range(n_sh)]

        enc_js   = [min(numOfScales, numOfScales - l) for l in range(half)]
        n_dec    = n_layers - half - 1
        full_mask = mask(numOfScales)   # all subbands

        # Encoder: masked (progressive coarse-to-fine)
        self.enc_blocks = nn.ModuleList([
            MaskedShearletConv(C, Psi, mask(j)) for j in enc_js])

        # Bottleneck: coarsest mask
        self.bot_block = MaskedShearletConv(C, Psi, mask(j_bot))

        # Decoder: UNMASKED — full shearlet basis, learns corrections
        self.dec_blocks = nn.ModuleList([
            MaskedShearletConv(C, Psi, full_mask) for _ in range(n_dec)])

        self.n_sh      = n_sh
        self.H, self.W = H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lifted = self.lift(x)                           # (B, C, H, W)

        # Global shortcut — learnable baseline
        # global_h = self.global_shortcut(x_lifted)         # (B, C, H, W)

        # ── Encoder: store (enc_output − global_h) as residual skip ──────────
        h = x_lifted
        skips = []
        for blk in self.enc_blocks:
            h = blk(h)
            skips.append(h - x_lifted)                    # residual skip

        # ── Bottleneck ────────────────────────────────────────────────────────
        h = self.bot_block(h)

        # ── Decoder: residual reconstruction (unmasked) ───────────────────────
        # h = residual_skip_i + dec_block(h)
        # → decoder learns correction to complete the residual representation
        for i, blk in enumerate(self.dec_blocks):
            skip_idx = len(skips) - 1 - i
            correction = blk(h)
            if 0 <= skip_idx < len(skips):
                h = skips[skip_idx] + correction
            else:
                h = correction

        # ── Add global baseline back + project ───────────────────────────────
        return self.project(self.act(h + x_lifted))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Training — end-to-end, same as USNO
# ─────────────────────────────────────────────────────────────────────────────

def train_cascade_model(model, train_loader, test_loader,
                         cfg: dict, device, ckpt_dir=None) -> dict:
    """End-to-end training — identical to train_standard in train_all.py."""
    import time
    from pathlib import Path
    import torch.optim as optim

    def _rl2(p, t):
        d=(p-t).reshape(p.shape[0],-1); t_=t.reshape(t.shape[0],-1)
        return (d.norm(dim=1)/(t_.norm(dim=1)+1e-8)).mean().item()

    def _mse(p, t):
        return ((p-t)**2).mean()/((t**2).mean().clamp(min=1e-8))

    @torch.no_grad()
    def _eval():
        model.eval(); tot=0.0
        for a, u in test_loader:
            a,u=a.to(device),u.to(device); tot+=_rl2(model(a),u)
        return tot/len(test_loader)

    opt   = optim.AdamW(model.parameters(),
                         lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'], eta_min=cfg['lr']*0.01)

    hist = {k: [] for k in ['epoch','train_rl2','test_rl2',
                              'epoch_time','cumtime','band','band_sigma']}
    cum=0.0; start_ep=1

    # Resume
    if ckpt_dir is not None:
        ckpt = Path(ckpt_dir)/'cascade.pt'
        if ckpt.exists():
            pl = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(pl['model_state'])
            if pl.get('opt_state'): opt.load_state_dict(pl['opt_state'])
            if pl.get('sched_state'): sched.load_state_dict(pl['sched_state'])
            hist    = pl['history']
            cum     = sum(hist['epoch_time'])
            start_ep= hist['epoch'][-1]+1
            print(f"  Resumed cascade from epoch {start_ep-1}")

    print(f"\n  [CascadeUSNO] {model.param_count():,} params  "
          f"epochs {start_ep}→{cfg['epochs']}")

    for ep in range(start_ep, cfg['epochs']+1):
        t0=time.time(); model.train(); ep_loss=0.0
        for a,u in train_loader:
            a,u=a.to(device),u.to(device)
            opt.zero_grad(); pred=model(a)
            _mse(pred,u).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); ep_loss+=_rl2(pred.detach(),u)
        sched.step()
        elapsed=time.time()-t0; cum+=elapsed
        tr=ep_loss/len(train_loader); te=_eval()
        hist['epoch'].append(ep); hist['train_rl2'].append(tr)
        hist['test_rl2'].append(te); hist['epoch_time'].append(elapsed)
        hist['cumtime'].append(cum); hist['band'].append(0)
        hist['band_sigma'].append(tr)
        if ep%cfg['log_every']==0 or ep==1:
            print(f"    ep {ep:4d}  train={tr:.5f}  test={te:.5f}  t={elapsed:.1f}s")
        if ckpt_dir and (ep%cfg.get('ckpt_every',10)==0 or ep==cfg['epochs']):
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'model_state':model.state_dict(),
                        'opt_state':opt.state_dict(),
                        'sched_state':sched.state_dict(),
                        'history':hist, 'extra':{}},
                       Path(ckpt_dir)/'cascade.pt')

    print(f"  [CascadeUSNO] Best relL2 = {min(hist['test_rl2']):.5f}")
    return hist