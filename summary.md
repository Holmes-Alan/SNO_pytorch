# Shock-bubble coarse→fine operator learning — experiment summary

Two experiments on the same dataset and the same four operator architectures.
They differ only in **what map the operator is asked to learn**.

| | Experiment 1 — `same` | Experiment 2 — `sr` |
|---|---|---|
| Map | 129² → 129² | 129² → 257² |
| Input | coarse simulation | bicubic upsample of coarse |
| Target | `fine[:, ::2, ::2]` | fine simulation |
| Learns | discretisation-error correction | correction **+** super-resolution |
| Free baseline | do nothing (copy input) | bicubic |
| Status | complete — trained Aug 19, evaluated Aug 19 | complete — trained Aug 20, evaluated Aug 21 |

**Research question.** Does a shearlet basis (SNO/USNO/Cascade) represent shock
fronts better than a Fourier basis (FNO) for operator learning on discontinuous
solutions? Experiment 1 isolates that question with no confounders; experiment 2
asks whether the answer survives a resolution change.

---

## 1. Data

`/raid/data/zhisong_liu/SNO/shock_bubble_sr` (mounted at `/workspace/data/SNO/…`
inside the training container). 20 Latin-hypercube samples of the compressible
Euler shock-bubble interaction, each simulated **twice from identical initial
data**:

- `coarse` — MFEM `-r 5` → 129×129 nodes
- `fine` — MFEM `-r 6` → 257×257 nodes

Because 257 = 2·129 − 1, the coarse nodes are *exactly* the even-indexed fine
nodes: `fine[:, ::2, ::2]` lives on the coarse grid with **zero interpolation**.
At t=0 the two runs agree to machine zero; all later divergence is pure
discretisation error. That is the signal the operator learns.

**Channels** (conserved variables, MFEM ordering, `[channel, y, x]`):
`0 rho`, `1 rho*u`, `2 rho*v`, `3 E`. Magnitudes differ by orders of magnitude and
the momenta are near-zero over most of the domain, so per-channel standardisation
is mandatory.

**Snapshots.** 24 per sample. `same` drops t=0 (degenerate: input == target to
machine zero, a free identity example). `sr` keeps it (bicubic upsampling of the
sharp initial interface is genuinely lossy).

**Split — by sample, never by snapshot**, since the 24 snapshots of one sample
share an initial condition. Near-duplicate pairs (07↔16, 04↔10, 03↔15, 14↔19)
are all kept inside `train`; val/test get the most isolated samples.

```
train [0,1,3,4,6,7,9,10,13,14,15,16,17,19]   val [5,11,18]   test [2,8,12]
```

Re-verify leakage for any other split with `python3 shock_dataset.py --check_split`.

---

## 2. Input / output contract

Both experiments predict the **residual to the input**, not the field:

```
x = (x_phys − mean) / std                     network input,  (4, N, N)
y = (y_phys − x_phys) / res_std               network target, (4, N, N)
pred_phys = x_phys + net(x) · res_std         inverse
```

`mean`, `std`, `res_std` are per-channel scalars computed over the **train split
only** (`results/shock/stats_{task}.json`) and shared by train/val/test — the only
leak-free choice. Residual targeting makes "do nothing" an explicit, free baseline
that is logged every validation epoch, so a model that fails to beat it is
immediately visible.

Set `--target direct` to regress the field itself instead.

**Patching.** `same` trains on whole 129² frames. `sr` at full 257² needs ~10+ GB
per block under autograd, so it trains on **128² random crops** (4 per frame) and
is evaluated by overlapping sliding window (stride 64, Gaussian blend) in
`predict_frame`. Crops never wrap the domain.

---

## 3. Models

Four operators, all FFT-based, built by `shock_common.build_model`. Common shell:
`Lift (1×1 conv) → blocks → Project (1×1 → GELU → 1×1)`, `hidden=32`.

| Model | Idea | Params |
|---|---|---|
| **FNO** | Per-block spectral conv on a `k_max × k_max` block of Fourier modes (2 non-redundant `rfft2` quadrants, real+imag) + pointwise bypass. `n_blocks=4`. | 267,716 |
| **SNO** | Same block structure, but the spectral mix is over the **29 shearlet subbands** of a cone-adapted Meyer filter bank (exact tight frame, roundtrip error < 1e-13, verified pixel-identical to FFST). Each block: shearlet analysis → learned per-subband channel mix → synthesis + pointwise bypass. `n_blocks=4`. | 124,356 |
| **USNO** | U-Net scale schedule over SNO-style blocks: encoder blocks see progressively fewer scales, bottleneck sees only the coarsest, decoder climbs back with spatial skip connections. Every block has its own FFT pair and spatial bypass (iterative refinement + cross-scale mixing). `n_layers=5`. | 101,924 |
| **Cascade USNO** | USNO with (a) **unmasked** decoder blocks, (b) a global shortcut `Lift(x) → conv` acting as a learnable baseline, (c) encoder skips storing the **residual to that shortcut**, so the decoder learns what each level captured *beyond* the baseline. `n_layers=5`. | 115,204 |

**Periodicity handling (`PadWrap`, `--pad 16`).** All four treat the domain as
periodic, but these fields are not — the walls reflect and the driver gas occupies
x < 0.1, so wrapping x=1 back to x=0 creates a discontinuity at ~100× (rho) and
~139× (E) the mean interior gradient. Every operator is therefore built for
`n + 2·pad` and wrapped in reflect-pad → operator → crop, so the wrap the FFT sees
is a mirror of the interior. Consequence: **the grid size is baked into the model
at construction** (the shearlet bank is a registered buffer), so a checkpoint
cannot be re-instantiated at a different resolution.

- `same`: 129 + 32 = **161²**, J=3 scales → n_sh=29
- `sr`: 128 + 32 = **160²**, J=3 scales → n_sh=29

**FNO bandwidth control — required for the claim to mean anything.**
`matched_k_max(n_sh) = round(sqrt(n_sh/2)) = 4` equalises the *spectral parameter
count* against SNO, but it also leaves FNO with a 4×4 block of Fourier modes out
of 128, which cannot resolve a shock. "SNO beats FNO" at matched k_max alone is
attributable to bandwidth, not to shearlet anisotropy. So a second FNO is trained
at `--k_max 32 --tag fno_wide` (16,782,788 params — 135× SNO). Only if SNO also
beats *that* is the result about the basis.

---

## 4. Evaluation

Full-frame inference on the held-out test samples, no patch-level scores (patch
metrics are optimistic — they never test consistency across seams). Loss during
training is a self-normalising MSE, `((p−t)²).mean() / (t²).mean()`, so channel
weighting is uniform; every **logged** metric is converted back to physical units
first, so numbers are comparable across models and tasks.

| Metric | What it measures |
|---|---|
| `rl2` | Global relative L2. Dominated by the large smooth regions — every method scores well. Reported for completeness, **not the headline**. |
| `rl2_highpass` | Relative L2 above 0.25 × Nyquist. ~6× more discriminating than global `rl2` on this data. **This is the fine-detail metric.** |
| `rl2_front` | Relative L2 on the top 10% of \|∇ρ\| — i.e. on the shock fronts. Mask comes from the **target**, so it is identical for every method. |
| `skill` | `1 − rl2 / baseline_rl2`. Fraction of the coarse solver's error actually removed. 0 = did nothing, 1 = perfect. |
| `mass_err` | Relative error in total mass. Conservation sanity check. |

Plus a radially averaged power spectrum per method (`figures/shock/shock_spectrum.png`):
a model that smooths shock structure away falls below the truth at high
wavenumber, visible directly rather than hidden in a scalar.

**Do-nothing baseline on the test split** (the number to beat):

| Task | frames | `rl2` | `rl2_highpass` | `rl2_front` |
|---|---|---|---|---|
| `same` (copy coarse) | 69 | 0.10328 | 0.60980 | 0.19163 |
| `sr` (bicubic) | 72 | 0.10131 | 0.70877 | 0.18260 |

---

## 5. Results — Experiment 1 (`same`), test split, 69 frames

200 epochs, AdamW lr 1e-3 → cosine to 1e-5, wd 1e-4, batch 8, grad clip 1.0.

| Method | params | `rl2` ↓ | `rl2_highpass` ↓ | `rl2_front` ↓ | `skill` ↑ | `mass_err` |
|---|---:|---:|---:|---:|---:|---:|
| do nothing | — | 0.10328 | 0.60980 | 0.19163 | 0.000 | 0.00044 |
| FNO (k=4) | 267,716 | 0.07435 | 0.36201 | 0.11809 | 0.279 | 0.00490 |
| **SNO** | 124,356 | **0.04061** | **0.21642** | **0.06469** | **0.602** | 0.00181 |
| Cascade USNO | 115,204 | 0.04186 | 0.21827 | 0.06712 | 0.590 | 0.00248 |
| USNO | 101,924 | 0.04324 | 0.22943 | 0.07017 | 0.577 | 0.00204 |
| FNO wide (k=32) | 16,782,788 | *not evaluated* | | | | |

**Reading.** All three shearlet models remove ~58–60% of the coarse solver's error;
param-matched FNO removes 28% with 2.2× more parameters. The gap widens on the
fine-detail metrics: SNO's high-pass error is 0.216 vs FNO's 0.362, and shock-front
error 0.065 vs 0.118. SNO, USNO and Cascade are within ~6% of each other — the
architecture *around* the shearlet transform matters much less than the transform
itself.

**Bandwidth control (val split, not yet run through `test_shock.py`).** Best
`val_rl2` over 200 epochs:

| | FNO k=4 | FNO k=32 | SNO | Cascade | USNO |
|---|---:|---:|---:|---:|---:|
| best `val_rl2` | 0.04941 | 0.08530 | 0.02245 | 0.02275 | 0.02339 |
| final `val/skill` | — | 0.017 | 0.746 | 0.743 | 0.736 |

Wide-band FNO ends at skill 0.017 — essentially *doing nothing* — and is worse
than the 4-mode FNO. **Do not report this as "bandwidth doesn't help".** 16.8M
params at lr 1e-3 on 322 training frames is far more likely an optimisation /
capacity-vs-data failure than evidence about the basis. It needs a lower LR and/or
weight decay sweep before it can serve as the control the claim requires.

---

## 6. Results — Experiment 2 (`sr`), test split, 72 frames

200 epochs, AdamW lr 1e-3 → cosine to 1e-5, wd 1e-4, **batch 4**, 128² random
crops (4 per frame). Evaluated **full-frame at 257²** by overlapping sliding
window (stride 64, Gaussian blend), so these numbers also test patch-seam
consistency — a strictly harder evaluation than experiment 1, which trains and
evaluates on whole frames.

Artifacts: `results/shock_sr/`, `checkpoints/shock_sr/`, `figures/shock_sr/`.
`fno_wide` was **not** run for this task.

| Method | params | `rl2` ↓ | `rl2_highpass` ↓ | `rl2_front` ↓ | `skill` ↑ | `mass_err` |
|---|---:|---:|---:|---:|---:|---:|
| bicubic (do nothing) | — | 0.10131 | 0.70877 | 0.18260 | 0.000 | 0.00074 |
| FNO (k=4) | 267,716 | 0.06670 | 0.40849 | 0.11284 | 0.346 | 0.00361 |
| **USNO** | 101,924 | **0.03918** | **0.24742** | **0.06550** | **0.614** | 0.00316 |
| SNO | 124,356 | 0.03988 | 0.25837 | 0.06683 | 0.608 | 0.00327 |
| Cascade USNO | 115,204 | 0.04019 | 0.26272 | 0.06796 | 0.604 | 0.00374 |

Best `val_rl2` over training: FNO 0.04225 (ep 199), USNO 0.02366 (ep 194),
SNO 0.02402 (ep 179), Cascade 0.02476 (ep 199). All four were still improving or
flat at 200 epochs; none diverged.

### Cross-experiment comparison

| | `same` `rl2` | `sr` `rl2` | `same` hp | `sr` hp | `same` skill | `sr` skill |
|---|---:|---:|---:|---:|---:|---:|
| do nothing | 0.10328 | 0.10131 | 0.60980 | 0.70877 | 0.000 | 0.000 |
| FNO | 0.07435 | 0.06670 | 0.36201 | 0.40849 | 0.279 | 0.346 |
| SNO | 0.04061 | 0.03988 | 0.21642 | 0.25837 | 0.602 | 0.608 |
| USNO | 0.04324 | 0.03918 | 0.22943 | 0.24742 | 0.577 | 0.614 |
| Cascade | 0.04186 | 0.04019 | 0.21827 | 0.26272 | 0.590 | 0.604 |

**Reading.**

1. **The result transfers.** The shearlet family removes ~60% of the input error in
   both tasks; FNO removes 28% (`same`) / 35% (`sr`) with 2.2× more parameters. The
   FNO-vs-shearlet gap is the same size on both maps, so it is not an artifact of
   the no-resolution-change setting.
2. **Adding super-resolution costs almost nothing.** Every model's global `rl2` is
   flat or slightly *better* on `sr` despite the harder map and the sliding-window
   evaluation. The two error sources (discretisation error, bicubic interpolation
   error) are not additive in practice — the coarse solver's error dominates both.
3. **High-pass error rises for everyone** (e.g. SNO 0.216 → 0.258) while the
   baseline's rises too (0.610 → 0.709): bicubic destroys high-frequency content
   the `same` task never had to reconstruct. Relative to their own baseline the
   models actually do slightly *better* here.
4. **USNO edges ahead of SNO on `sr`** (0.03918 vs 0.03988, +1.8%) after losing to
   it on `same`. With 3 test samples this is noise, not a finding. The honest
   statement across both experiments is that SNO / USNO / Cascade are
   indistinguishable and all clearly beat FNO.

### Known issues, fixed

The first SR attempt (Aug 19) failed: `train_shock.py` auto-resumes from
`ckpt_dir/<method>/last.ckpt` and the run name carries no task, so `same` and `sr`
shared `checkpoints/shock/<method>/`. SNO/USNO/Cascade died on a `Psi` shape
mismatch (161² vs 160²); FNO, being resolution-independent, silently resumed at
epoch 199/200 and "finished" in one epoch. Fixed in two places, and the successful
Aug 20 run used separate `--out_dir`/`--ckpt_dir`:

- `train_shock.py` validates a resume candidate's `task/patch/pad/hidden/n_blocks/
  n_layers/n_scales/k_max` against the current config and aborts with an
  explanation instead of splicing two runs together.
- `test_shock.py` rebuilds each operator at the geometry recorded **in its
  checkpoint** rather than at `--patch or grid_size(task)`.

Residue from that failed attempt, in the *old* directory only:
`checkpoints/shock/fno/last.ckpt` is a hybrid (`same` weights, `task: sr`
hyperparameters) and `results/shock/fno_history.csv` was overwritten by the
one-epoch run. The `same`-task FNO history survives in
`results/shock/tensorboard/fno/events.out.tfevents.1787120357.*`.

### Commands used

```bash
python3 train_shock.py --task sr --patch 128 --batch 4 \
    --out_dir results/shock_sr --ckpt_dir checkpoints/shock_sr

python3 test_shock.py  --task sr --patch 128 \
    --out_dir results/shock_sr --ckpt_dir checkpoints/shock_sr \
    --fig_dir figures/shock_sr
```

## 7. Caveats

- **`rl2_rho_v` is meaningless — ignore that column.** Transverse momentum is
  near-zero over most of the domain, so its per-channel relative L2 blows up
  (57–89 across methods on `same`). On `sr` it reaches ~10¹⁰ for a specific
  reason: `sr` keeps the t=0 frame, where `rho_v ≡ 0` *identically*, so the
  denominator falls back to the 1e-12 guard. The median over `sr` frames is
  0.142, which is the sensible value. Global `rl2` is unaffected — it takes the
  norm over all four channels jointly and is dominated by rho and E.
- **Small test set.** 3 samples (69–72 frames). Frames within a sample are highly
  correlated, so the effective n is closer to 3 than 70 — the SNO/USNO/Cascade
  ordering is not resolvable at this sample size.
- **The bandwidth control is not yet usable** (see §5). Until wide FNO is trained
  competently, "SNO beats FNO because of the basis" is unsupported.
- **`fno_wide` was never run through `test_shock.py`** — add it via
  `--names fno sno usno cascade fno_fno_wide`.
- Mass error rises for every learned model relative to the baseline (0.0004 →
  0.002–0.005). Nothing in the loss enforces conservation.
- **`val/rl2_highpass` in the history CSVs is garbage for `sr`** (values up to
  1e12). It is computed per 128² *patch*, and a smooth patch has near-zero energy
  above 0.25 Nyquist, so the ratio explodes. Only the test-time, full-frame
  `rl2_highpass` in §5/§6 is meaningful.
- **Full-frame validation never reached the history CSVs.** `HistoryCSVCallback`
  is registered before `FullFrameValCallback` in `train_shock.py`, so it snapshots
  `callback_metrics` before the `val_full/*` keys are logged. Those numbers exist
  only in stdout and TensorBoard. Harmless, but swap the callback order if you
  want them in the CSV.

## 8. File map

| File | Role |
|---|---|
| `shock_dataset.py` | pair construction, splits, normalisation stats, leakage check |
| `shock_common.py` | model factory, `PadWrap`, metrics, sliding-window inference |
| `train_shock.py` | Lightning training for both tasks |
| `test_shock.py` | full-frame evaluation + figures (no Lightning dependency) |
| `fno_urban.py` / `sno_urban.py` / `usno_urban.py` / `cascade_usno_urban.py` | the four operators |
