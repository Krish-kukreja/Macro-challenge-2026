# DREAMPlace-DiffOpt: Macro Placement via Differentiable Proxy Optimization

## Method Overview

A four-stage macro placement pipeline that combines GPU-accelerated analytical placement with direct optimization of the TILOS proxy cost formula.

### Stage 1: DREAMPlace Global Placement
- GPU-accelerated electrostatic-field-based analytical placer
- Multi-mode strategy: runs default + congestion-aware configurations
- Best-proxy checkpoint selection (picks optimal iteration, not just final)
- Adaptive target density based on benchmark characteristics

### Stage 2: Differentiable Proxy Optimizer
- Directly optimizes the TILOS proxy formula using PyTorch autograd
- **Exact rectangular overlap density model** matching TILOS computation
- **Soft L-routing congestion model** with sigmoid-based tile indicators
- **Soft ABU5** via detached quantile threshold
- Progressive overlap penalty curriculum (0 → 2000) for maximum exploration
- Iterative rounds: 3× coarse pass (lr=0.005) → cleanup → fine pass (lr=0.0008)
- Multi-start: runs from DREAMPlace output + CT initial placement + random jittered starts

### Stage 3: ABU5 Coordinate Descent
- Post-optimization pass using the **actual TILOS evaluator** (not approximation)
- Ranks macros by net degree (high-degree = most congestion impact)
- Tries 8 directional moves per macro (cardinal + diagonal), accepts only moves that reduce real proxy
- Guaranteed improvement: every accepted move reduces the actual score
- Step decay when no improvement; time-budgeted (300s)

### Stage 4: Overlap Cleanup
- 4-stage pipeline guaranteeing zero overlaps:
  1. Min-displacement legalization (pairwise repulsion)
  2. Jiggle + retry (random perturbation for trapped pairs)
  3. Force-push (greedy unconditional translation)
  4. Nuclear fallback (CT initial positions)

## Key Innovation

The differentiable optimizer eliminates the **objective mismatch** between DREAMPlace (which optimizes HPWL + electrostatic density) and the TILOS evaluator (which scores WL + density + ABU5 congestion). By directly optimizing what gets scored, we achieve 5-10% additional improvement over DREAMPlace alone.

## Results

| Metric | Value |
|--------|-------|
| Average proxy cost (17 IBM benchmarks) | 1.1994 |
| All benchmarks valid | ✓ (0 overlaps) |
| Runtime per benchmark | ~51 min (RTX 4050) / ~15 min (RTX 6000 Ada) |

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

**Result:** Full pipeline → **1.1994 average proxy**. Each component contributes measurably.

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

## How to Run

### Option 1: Docker (Recommended for Judges)

This is the easiest way to run the submission. The Dockerfile handles all dependencies including Python 3.10 (required for our compiled .so files).

```bash
# Clone the repo
git clone https://github.com/Krish-kukreja/Macro-challenge-2026.git
cd Macro-challenge-2026

# Initialize the TILOS submodule (needed for evaluation)
git submodule update --init external/MacroPlacement

# Build the Docker image (downloads ~4GB, takes ~5 min)
docker build -t dreamplace-diffopt -f submissions/analytical_placer/Dockerfile .

# Run on all 17 IBM benchmarks (--network none enforced at runtime)
docker run --gpus all --network none \
  dreamplace-diffopt \
  /submission/placer.py --all

# Run on a single benchmark
docker run --gpus all --network none \
  dreamplace-diffopt \
  /submission/placer.py -b ibm01
```

### Option 2: Local (Development)

Requires Python 3.10, CUDA 11.8+, and an NVIDIA GPU.

```bash
# Clone and setup
git clone https://github.com/Krish-kukreja/Macro-challenge-2026.git
cd Macro-challenge-2026
git submodule update --init external/MacroPlacement

# Install dependencies (Python 3.10 required for .so compatibility)
pip install -e .

# Run on single benchmark
python -m macro_place.evaluate submissions/analytical_placer/placer.py -b ibm01

# Run on all 17 benchmarks
python -m macro_place.evaluate submissions/analytical_placer/placer.py --all

# Run sweep with JSON output
python dreamplace_integration/sweep.py --out results.json
```

### Expected Output

```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    0.8690    1.3166    0.9976    +34.0%      +12.9%         0
   ibm02    1.2133    1.9072    1.8370    +36.4%      +33.9%         0
   ...
     AVG    ~1.19     2.1251    1.4578    +44.0%      +18.4%         0
```

Each benchmark takes ~50 min on RTX 4050 or ~15 min on RTX 6000 Ada.

## Requirements

- **Python 3.10** (our DREAMPlace .so files are compiled for cpython-310)
- **PyTorch 2.5.1+** with CUDA
- **NVIDIA GPU** (RTX 4050+ recommended, needs CUDA 11.8+ driver)
- See `requirements.txt` for full list

The Dockerfile at `submissions/analytical_placer/Dockerfile` handles all of this automatically.

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

## Files

```
submissions/analytical_placer/
├── placer.py                    ← Main algorithm (DreamplacePlacer class)
├── Dockerfile                   ← Python 3.10 runtime environment
└── dreamplace_bundle/           ← Self-contained DREAMPlace (93MB)

dreamplace_integration/
├── diff_proxy_optimizer.py      ← Differentiable TILOS proxy optimizer
├── abu5_shifter.py              ← Coordinate descent on real ABU5
├── pb2bookshelf.py              ← TILOS → Bookshelf format converter
├── post_legalize.py             ← 4-stage overlap cleanup pipeline
├── check_overlaps.py            ← Overlap detection
└── sweep.py                     ← Benchmark sweep harness
```

## Author

Krish Kukreja
