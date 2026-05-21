# DREAMPlace-DiffOpt: Macro Placement via Differentiable Proxy Optimization

## Method Overview

A three-stage macro placement pipeline that combines GPU-accelerated analytical placement with direct optimization of the TILOS proxy cost formula.

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
- Iterative rounds: coarse pass (lr=0.005) → cleanup → fine pass (lr=0.0008)
- Multi-start: runs from DREAMPlace output AND CT initial placement

### Stage 3: ABU5 Coordinate Descent
- Post-optimization pass using the **actual TILOS evaluator** (not approximation)
- Ranks macros by net degree (high-degree = most congestion impact)
- Tries 8 directional moves per macro, accepts only moves that reduce real proxy
- Guaranteed improvement: every accepted move reduces the actual score

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
| Average proxy cost (17 IBM benchmarks) | ~1.19 |
| All benchmarks valid | ✓ (0 overlaps) |
| Runtime per benchmark | ~50 min (RTX 4050) / ~15 min (RTX 6000 Ada) |

## How to Run

```bash
# Install dependencies
pip install -e .

# Run on single benchmark
python -m macro_place.evaluate submissions/analytical_placer/placer.py -b ibm01

# Run on all 17 benchmarks
python -m macro_place.evaluate submissions/analytical_placer/placer.py --all

# Run sweep (saves results to JSON)
python dreamplace_integration/sweep.py --out results.json
```

## Requirements

- Python 3.10 (for DREAMPlace .so compatibility)
- PyTorch 2.5.1+ with CUDA
- NVIDIA GPU (RTX 4050+ recommended)
- See `requirements.txt` for full list

A Dockerfile is provided at `submissions/analytical_placer/Dockerfile` for reproducible builds.

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
  Differentiable Proxy Optimizer (coarse + fine passes)
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
