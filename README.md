# DREAMPlace-DiffOpt: Macro Placement via Differentiable Proxy Optimization

> **Submission for the Partcl/HRT Macro Placement Challenge 2026**
>
> Team: Krish Kukreja | [Competition Page](https://github.com/partcleda/partcl-macro-place-challenge)

## Results

| Metric | Value |
|--------|-------|
| **Average proxy cost** | **1.1994** (17 IBM benchmarks) |
| All benchmarks valid | ✓ (0 overlaps on all 17) |
| Runtime per benchmark | ~51 min (RTX 4050) / ~15 min (RTX 6000 Ada) |
| vs RePlAce baseline (1.4578) | **17.7% better** |
| vs SA baseline (2.1251) | **43.6% better** |

## Method Overview

A four-stage macro placement pipeline that combines GPU-accelerated analytical placement with direct optimization of the TILOS proxy cost formula.

### Stage 1: DREAMPlace Global Placement
- GPU-accelerated electrostatic-field-based analytical placer (open-source, BSD license)
- Multi-mode strategy: runs default + congestion-aware configurations
- Best-proxy checkpoint selection (picks optimal iteration, not just final)
- Adaptive target density based on benchmark characteristics

### Stage 2: Differentiable Proxy Optimizer (Key Innovation)
- Directly optimizes the TILOS proxy formula using PyTorch autograd
- **Exact rectangular overlap density model** matching TILOS computation
- **Soft L-routing congestion model** with sigmoid-based tile indicators (temp=0.15)
- **Soft ABU5** via detached quantile threshold
- Progressive overlap penalty curriculum (0 → 2000) for maximum exploration
- Iterative: 3 rounds of coarse pass (lr=0.005) → cleanup → fine pass (lr=0.0008)
- Multi-start: runs from DREAMPlace output + CT initial placement + random jittered starts

### Stage 3: ABU5 Coordinate Descent
- Post-optimization pass using the **actual TILOS evaluator** (not approximation)
- Ranks macros by net degree (high-degree = most congestion impact)
- Tries 8 directional moves per macro (cardinal + diagonal)
- Accepts only moves that reduce real proxy AND don't create overlaps
- Step decay when no improvement; time-budgeted (300s)

### Stage 4: Overlap Cleanup
- 4-stage pipeline guaranteeing zero overlaps:
  1. Min-displacement legalization (iterative pairwise repulsion)
  2. Jiggle + retry (random perturbation for geometrically trapped pairs)
  3. Force-push (greedy unconditional translation)
  4. Nuclear fallback (CT initial positions — never triggered in practice)

## Key Innovation

The differentiable optimizer eliminates the **objective mismatch** between DREAMPlace (which optimizes HPWL + electrostatic density) and the TILOS evaluator (which scores WL + density + ABU5 congestion). By directly optimizing what gets scored, we achieve 5-10% additional improvement over DREAMPlace alone.

**Why this works:** DREAMPlace's internal loss doesn't include congestion at all. The TILOS proxy weights congestion at 0.5×. Our differentiable optimizer models congestion via soft L-routing demand accumulation with sigmoid-based range indicators, box-filter smoothing (matching TILOS smooth_range=2), and soft ABU5 selection — all fully differentiable.

## Development Journey

This project evolved through rapid iteration over ~10 days. Here's how we got from a broken placer to a competitive submission:

### Phase 1: Getting DREAMPlace Working (Days 1-3)

**Starting point:** The competition's example greedy placer scores ~2.21. We wanted to use DREAMPlace (GPU-accelerated analytical placer) but it doesn't natively support the TILOS benchmark format.

- **Built DREAMPlace from source** in WSL2 Ubuntu 22.04 against CUDA 11.8 (RTX 4050, sm_89)
- **Wrote a format converter** (`pb2bookshelf.py`) to translate TILOS `.pb.txt` + `.plc` files into Bookshelf format that DREAMPlace understands
- **Patched DREAMPlace** for NumPy 2.0 compatibility, WSL2 CUDA detection issues, and pin_pos fallback bugs
- **Built the overlap cleanup pipeline** — DREAMPlace produces overlap-free standard cell placements but not macro placements, so we needed a custom legalizer

**Result:** DREAMPlace alone → ~1.33 average proxy. Already beating RePlAce (1.46).

### Phase 2: Differentiable Optimizer (Days 4-6)

**Problem identified:** DREAMPlace optimizes its own internal objective (HPWL + electrostatic density). The TILOS proxy formula is different — especially the congestion term (ABU5 of L-routed demand). There's an objective mismatch.

- **First attempt:** Gaussian splatting for density → diverged because it didn't match TILOS's actual computation
- **Fix:** Switched to exact rectangular overlap density (what TILOS actually computes) → perfect correlation
- **Added soft L-routing congestion** with sigmoid-based range indicators, matching TILOS's star decomposition
- **Progressive overlap curriculum** (0 → 2000): let macros overlap freely early for exploration, then force them apart
- **Multi-pass strategy:** coarse (lr=0.005) then fine (lr=0.0008) for deeper convergence

**Result:** DREAMPlace + diff_opt → ~1.22 average proxy. 5-8% improvement from the optimizer alone.

### Phase 3: Iterative Refinement (Days 7-8)

**Insight:** Running the optimizer once leaves room for improvement. Each round of optimize → cleanup → optimize again finds a slightly better local minimum.

- **Iterative rounds:** 3 rounds of (coarse + fine + cleanup) instead of just one
- **Multi-start:** Also try optimizing from the CT initial placement and random jittered positions — different starting points find different basins
- **Checkpoint selection:** Instead of using DREAMPlace's final iteration, scan all checkpoints and pick the one with best proxy after cleanup

**Result:** Iterative + multi-start → ~1.20 average proxy.

### Phase 4: Real TILOS Coordinate Descent (Days 9-10)

**Insight:** Our differentiable congestion is an approximation. The real TILOS evaluator is the ground truth. Why not use it directly?

- **ABU5 coordinate descent:** For top-10 macros by net degree, try 8 directional moves and accept only moves that reduce the REAL proxy
- **Guaranteed improvement:** Every accepted move makes the actual score better (no approximation error)
- **Time-budgeted:** Runs for up to 300s, with step decay when no improvement found

**Result:** Full pipeline → **~1.19 average proxy**. Each component contributes measurably.

### What We Tried That Didn't Work

- **Gaussian splatting for density** — diverged because it doesn't match TILOS's computation
- **RUDY congestion model** — TILOS uses L-routing, not uniform bounding-box distribution
- **Very high overlap penalty from the start** — constrains exploration too early, worse final results
- **Single-pass optimization** — iterative rounds consistently find better solutions
- **Fine grid DREAMPlace (1024 bins)** — slower but not consistently better than default
- **Orientation search (N/FN/FS/S flips)** — no improvement because our optimizer uses macro centers, not pin offsets

### Progression Summary

| Version | Avg Proxy | Key Change |
|---------|-----------|------------|
| Greedy baseline | 2.21 | Competition example |
| Bug fixes + legalizer | 1.47 | Basic DREAMPlace working |
| DREAMPlace integration | 1.33 | Format converter + cleanup pipeline |
| Checkpoint selection | 1.32 | Pick best iteration, not just final |
| Multi-mode strategy | 1.28 | default + congestion_aware configs |
| Differentiable optimizer | 1.22 | Direct TILOS proxy optimization |
| Iterative rounds | 1.20 | 3 rounds of coarse + fine |
| ABU5 coordinate descent | **1.1994** | Real TILOS evaluator moves |

## Per-Benchmark Results

| Benchmark | Proxy | WL | Density | Congestion | Overlaps |
|-----------|-------|-----|---------|------------|----------|
| ibm01 | 0.8690 | 0.068 | 0.485 | 1.118 | 0 |
| ibm02 | 1.2133 | 0.078 | 0.518 | 1.754 | 0 |
| ibm03 | 1.0904 | 0.085 | 0.561 | 1.451 | 0 |
| ibm04 | 1.1371 | 0.077 | 0.500 | 1.620 | 0 |
| ibm06 | 1.3832 | 0.074 | 0.501 | 2.118 | 0 |
| ibm07 | 1.1337 | 0.070 | 0.514 | 1.614 | 0 |
| ibm08 | 1.3318 | 0.081 | 0.503 | 1.998 | 0 |
| ibm09 | 0.8905 | 0.058 | 0.506 | 1.159 | 0 |
| ibm10 | 1.1070 | 0.059 | 0.517 | 1.578 | 0 |
| ibm11 | 0.9680 | 0.062 | 0.500 | 1.312 | 0 |
| ibm12 | 1.2736 | 0.072 | 0.505 | 1.898 | 0 |
| ibm13 | 1.1166 | 0.057 | 0.564 | 1.556 | 0 |
| ibm14 | 1.2582 | 0.060 | 0.495 | 1.901 | 0 |
| ibm15 | 1.3805 | 0.064 | 0.505 | 2.128 | 0 |
| ibm16 | 1.2564 | 0.055 | 0.524 | 1.879 | 0 |
| ibm17 | 1.4836 | 0.064 | 0.453 | 2.386 | 0 |
| ibm18 | 1.4971 | 0.071 | 0.479 | 2.373 | 0 |

## How to Run

```bash
# Install dependencies
pip install -e .

# Run on single benchmark
python -m macro_place.evaluate submissions/analytical_placer/placer.py -b ibm01

# Run on all 17 benchmarks
python -m macro_place.evaluate submissions/analytical_placer/placer.py --all
```

### With Docker (recommended for judges)

```bash
# Clone with submodules (REQUIRED for benchmarks)
git clone --recursive https://github.com/Krish-kukreja/Macro-challenge-2026.git
cd Macro-challenge-2026

# If already cloned without --recursive:
# git submodule update --init external/MacroPlacement

# Build the image (~5 min, downloads Python 3.10 + PyTorch)
docker build -t dreamplace-diffopt .

# Run evaluation on all 17 benchmarks (--network none enforced)
docker run --gpus all --network none dreamplace-diffopt

# Run on a single benchmark
docker run --gpus all --network none dreamplace-diffopt -b ibm01
```

## Architecture

```
Input: TILOS benchmark (.pb.txt + initial.plc)
  │
  ├─→ DREAMPlace (default mode) ──→ checkpoint scan ──┐
  │                                                    │
  ├─→ DREAMPlace (congestion_aware) → checkpoint scan ─┤
  │                                                    │
  └─→ Pick best mode ─────────────────────────────────┘
         │
         ▼
  Differentiable Proxy Optimizer (3 rounds × coarse + fine)
         │
         ▼
  Multi-start: CT initial + 3 random jittered starts
         │
         ▼
  ABU5 Coordinate Descent (real TILOS evaluator)
         │
         ▼
  4-Stage Overlap Cleanup → 0 overlaps guaranteed
         │
         ▼
  Output: [num_macros, 2] placement tensor
```

## Technical Details

### Differentiable Proxy Formula

```
Loss = WL + 0.5 × Density + 0.5 × Congestion + overlap_weight × Overlap
```

- **WL:** Log-sum-exp HPWL approximation, normalized by canvas area
- **Density:** Exact rectangular overlap between macro bboxes and grid cells → soft top-10% (ABU10)
- **Congestion:** Soft L-routing (2-3 pins/net via star decomposition) → sigmoid range indicators → box-filter smoothing → soft ABU5
- **Overlap:** Progressive curriculum (0 → 2000) allowing free exploration before forcing convergence

### Why Each Component Matters

| Component | Avg Proxy | Contribution |
|-----------|-----------|--------------|
| DREAMPlace alone | ~1.32 | Density optimization via electrostatics |
| + Multi-mode + checkpoint | ~1.28 | Picks best DREAMPlace config |
| + Differentiable optimizer | ~1.22 | Directly optimizes TILOS proxy |
| + Iterative rounds (3×) | ~1.20 | Deeper convergence |
| + ABU5 coordinate descent | ~1.19 | Real TILOS moves, guaranteed improvement |
| + Multi-start (CT + random) | ~1.19 | Finds better basins on some benchmarks |

## Files

```
submissions/analytical_placer/
├── placer.py                    ← Main algorithm (DreamplacePlacer class)
├── Dockerfile                   ← Python 3.10 runtime environment
└── dreamplace_bundle/           ← Self-contained DREAMPlace (93MB)
    ├── dreamplace/              ← DREAMPlace package + compiled .so ops
    ├── lib/libcudart.so.11.0    ← CUDA runtime for forward compat
    ├── thirdparty/NCTUgr        ← Global router binary (statically linked)
    ├── _bundle_env.py           ← Environment setup helper
    ├── pb2bookshelf.py          ← TILOS → Bookshelf converter
    └── post_legalize.py         ← 4-stage overlap cleanup

dreamplace_integration/
├── diff_proxy_optimizer.py      ← Differentiable TILOS proxy optimizer
├── abu5_shifter.py              ← Coordinate descent on real ABU5
├── pb2bookshelf.py              ← TILOS → Bookshelf format converter
├── post_legalize.py             ← 4-stage overlap cleanup pipeline
├── check_overlaps.py            ← Overlap detection utility
└── sweep.py                     ← Benchmark sweep harness
```

## Compliance

- ✅ General algorithm (no per-benchmark hardcoding — `BENCHMARK_OVERRIDE = {}`)
- ✅ Zero overlaps on all benchmarks
- ✅ Under 1 hour per benchmark (~50 min on RTX 4050, ~15 min on eval server)
- ✅ Open-source tools only (DREAMPlace BSD, PyTorch BSD)
- ✅ TILOS evaluator used unmodified
- ✅ No macro rotation (N orientation only)
- ✅ No soft macro resizing
- ✅ Dockerfile provided for Python 3.10 compatibility

## Hardware

- **Development:** Windows 11, RTX 4050 6GB, 16GB RAM, WSL2 Ubuntu 22.04
- **Eval server:** AMD EPYC 9655P 16-core, RTX 6000 Ada 48GB, 100GB RAM

## See Also

- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — Full technical report with design decisions

## Author

**Krish Kukreja**
