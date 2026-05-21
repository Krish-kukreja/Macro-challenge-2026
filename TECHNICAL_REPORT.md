# Technical Report: DREAMPlace + Differentiable Proxy Optimizer

## Result

**Average proxy cost: ~1.19** across 17 ICCAD04 IBM benchmarks (v10)
- All 17 benchmarks: 0 overlaps
- 19% better than the 1.4738 baseline
- 18% better than RePlAce (1.4578)
- Runtime: ~50 min per benchmark on RTX 4050 laptop; ~15 min on eval server

## The Problem

Place 246-537 hard macros on a chip canvas to minimize:
```
Proxy = Wirelength + 0.5 × Density + 0.5 × Congestion
```
Subject to: zero overlaps between macros, all macros within canvas bounds.

## Our Approach: Three-Stage Pipeline

### Stage 1: DREAMPlace Global Placement

We use DREAMPlace (open-source GPU-accelerated analytical placer) as the initial placement engine. DREAMPlace optimizes wirelength + density using electrostatic-field-based gradient descent on GPU.

**Why DREAMPlace:** It produces high-quality global placements in ~100 seconds on GPU. Its density optimization is fundamentally better than simulated annealing or greedy approaches.

**Key modifications we made:**
- Built from source against CUDA 11.8 with RTX 4050 (sm_89) support
- Patched for NumPy 2.0 compatibility, WSL2 CUDA detection, pin_pos fallback
- Added adaptive `target_density` based on benchmark characteristics
- Added iteration checkpoint dumping for best-proxy selection
- Bundled as self-contained 93MB package with libcudart.so.11.0

**Multi-mode strategy:** We run DREAMPlace with two configurations per benchmark:
- `default`: standard settings (density_weight=8e-5, stop_overflow=0.10)
- `congestion_aware`: tighter convergence (density_weight=1.6e-4, stop_overflow=0.05)

We pick whichever mode produces the lower proxy after cleanup.

### Stage 2: Differentiable Proxy Optimizer (the key innovation)

After DREAMPlace, we fine-tune the placement using gradient descent directly on a differentiable approximation of the TILOS proxy formula.

**Why this matters:** DREAMPlace optimizes its own internal objective (HPWL + electrostatic density). The TILOS proxy formula is different - especially the congestion term (ABU5 of L-routed demand tiles). Our differentiable optimizer eliminates this objective mismatch by directly optimizing what gets scored.

**Implementation:**
- **Differentiable HPWL:** Log-sum-exp approximation of half-perimeter wirelength
- **Differentiable Density:** Exact rectangular overlap between macro bounding boxes and grid cells (matching TILOS computation exactly), with soft top-10% selection via detached quantile threshold
- **Differentiable Congestion:** Soft L-routing demand accumulation (2-3 pins per net via star decomposition) using sigmoid-based range indicators (temp=0.15), box-filter smoothing matching TILOS's smooth_range=2, and soft ABU5 via detached quantile threshold
- **Overlap penalty:** Progressive curriculum (0 early → 2000 late) allowing free exploration before forcing overlap-free convergence

**Optimizer:** Adam with cosine annealing learning rate schedule. Iterative multi-pass strategy:
- 3 rounds of: coarse pass (lr=0.005) → cleanup → fine pass (lr=0.0008) → cleanup
- Multi-start: runs from DREAMPlace output AND CT initial placement AND random jittered starts
- Picks the best result across all starts and rounds

### Stage 3: ABU5 Coordinate Descent (real TILOS evaluator)

After the differentiable optimizer, we run coordinate descent using the **actual TILOS evaluator** as the objective, not our approximation. This guarantees every accepted move reduces the real score.

- Ranks macros by net degree (high-degree macros have most impact on congestion)
- For top-10 macros, tries 8 directional moves (cardinal + diagonal)
- Accepts only moves that reduce the real TILOS proxy AND don't create overlaps
- Reduces step size when no improvement found; stops when step < 0.2 grid cells
- Time-budgeted to stay within the 1-hour limit

### Stage 4: 4-Stage Overlap Cleanup

Guarantees zero overlaps regardless of what the optimizer produces:
1. **Min-displacement legalize:** Iterative pairwise repulsion (gentle, avg 0.1μm displacement)
2. **Jiggle retry:** Random perturbation + re-legalize (handles geometrically trapped pairs)
3. **Force push:** Greedy unconditional translation (handles stubborn cases)
4. **Nuclear fallback:** Restore CT initial.plc positions (never triggered in practice)

## Per-Benchmark Results (v8)

```
Benchmark  Proxy   WL      Density  Congestion  vs CT Initial
ibm01      0.878   0.067   0.487    1.136       -15.4%
ibm02      1.234   0.076   0.525    1.791       -21.2%
ibm03      1.105   0.083   0.540    1.504       -16.6%
ibm04      1.162   0.077   0.512    1.658       -11.5%
ibm06      1.425   0.073   0.501    2.205       -14.0%
ibm07      1.136   0.069   0.515    1.619       -23.1%
ibm08      1.412   0.079   0.503    2.164       -3.7%
ibm09      0.910   0.057   0.516    1.190       -18.2%
ibm10      1.134   0.060   0.531    1.616       -15.4%
ibm11      0.988   0.061   0.501    1.353       -18.6%
ibm12      1.274   0.072   0.507    1.896       -21.6%
ibm13      1.115   0.056   0.562    1.558       -19.5%
ibm14      1.295   0.060   0.505    1.967       -18.7%
ibm15      1.419   0.063   0.500    2.213       -11.5%
ibm16      1.244   0.056   0.501    1.874       -16.6%
ibm17      1.502   0.064   0.447    2.429       -13.6%
ibm18      1.570   0.070   0.465    2.536       -12.3%
AVG        1.224                                 -16.0%
```

## Why Each Component Matters

| Component | Contribution to final result |
|-----------|------------------------------|
| DREAMPlace alone | Gets us to ~1.28 avg (density optimization) |
| Multi-mode + checkpoint | Gets us to ~1.28 (picks best DREAMPlace config) |
| Differentiable optimizer | Gets us to ~1.20 (directly optimizes TILOS proxy) |
| Iterative rounds (3×) | Gets us to ~1.19 (deeper convergence) |
| ABU5 coordinate descent | Additional -0.01 to -0.03 per benchmark (real TILOS moves) |
| Multi-start (CT + random) | Finds better basins on some benchmarks |
| Cleanup pipeline | Guarantees 0 overlaps (required for validity) |

## Technical Decisions

**Why not just use the differentiable optimizer from scratch (no DREAMPlace)?**
DREAMPlace provides a much better starting point than random initialization. The differentiable optimizer is a local optimizer, it refines a good solution but can't find one from scratch. DREAMPlace's electrostatic solver handles the global structure; our optimizer handles the fine-tuning.

**Why exact rectangular overlap for density (not Gaussian splatting)?**
We tried Gaussian splatting first, it caused the optimizer to diverge because the approximation didn't match TILOS's actual density computation. Switching to exact rectangular overlap (which is what TILOS computes) made the differentiable loss correlate perfectly with the real proxy.

**Why progressive overlap penalty (curriculum)?**
Starting with zero overlap penalty lets the optimizer explore freely, macros can temporarily overlap while finding better positions. The penalty ramps up over training, forcing the optimizer to resolve overlaps by the end. This produces better final placements than a constant high penalty (which constrains exploration too early).

**Why L-routing for congestion (not RUDY)?**
The TILOS evaluator uses L-shaped routing (horizontal then vertical) for 2-pin nets, not RUDY (uniform bounding-box distribution). Our differentiable congestion matches this exactly using sigmoid-based soft range indicators.

## Runtime Breakdown (per benchmark, RTX 4050)

```
DREAMPlace (2 modes × ~100s each):     ~200s
Checkpoint scan:                         ~60s
Differentiable optimizer (3 rounds):    ~1500s
ABU5 coordinate descent:                ~300s
Multi-start (CT + random):              ~600s
Cleanup:                                 ~5s
Total:                                  ~2500-3000s (42-50 min)
```

On the eval server (RTX 6000 Ada, 3-4× faster): ~12-15 min per benchmark. Well within the 1-hour limit.

## Compliance

- General algorithm (no per-benchmark hardcoding)
- Zero overlaps on all benchmarks
- Under 1 hour per benchmark (with margin)
- Open-source tools only (DREAMPlace BSD, PyTorch BSD)
- TILOS evaluator used unmodified
- No macro rotation (N orientation only)
- No soft macro resizing
- Dockerfile provided for Python 3.10 compatibility

## Hardware

- Development: Windows 11, RTX 4050 6GB, 16GB RAM, WSL2 Ubuntu 22.04
- Eval server: AMD EPYC 9655P 16-core, RTX 6000 Ada 48GB, 100GB RAM
