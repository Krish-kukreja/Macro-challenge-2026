"""
Multi-stage post-legalize pipeline for DREAMPlace output.

Stages, in order:
  1. min_displacement_legalize — iterative pairwise repulsion (gentle)
  2. jiggle_retry              — random small perturbation + retry, N attempts
  3. force_push                — unconditional minimum-translation per pair
  4. nuclear_fallback          — restore CT initial.plc positions for unresolved

Pipeline returns the first stage that produces a clean placement (zero
overlaps). All four stages run if needed; we never write a placement
that still has overlaps unless every stage failed (extreme fallback).

Public entry point used by sweep_dreamplace.py:
    clean_overlaps(pos, benchmark,
                   legal_max_sweeps, jiggle_attempts,
                   force_max_iterations, margin, verbose) -> (clean_pos, CleanupResult)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
import numpy as np
import torch

from dreamplace_integration.check_overlaps import check_overlaps, check_overlaps_torch


@dataclass
class CleanupResult:
    """Structured result from the cleanup pipeline."""
    stage_used: str = "none"          # 'stage1' / 'stage2' / 'stage3' / 'stage4' / 'failed' / 'already-clean'
    n_overlaps_in: int = 0
    n_overlaps_out: int = 0
    avg_displacement: float = 0.0
    max_displacement: float = 0.0
    timings_s: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ─── Stage 1: original iterative pairwise repulsion ───────────────────────

def min_displacement_legalize(
    pos: torch.Tensor,
    benchmark,
    max_sweeps: int = 300,
    margin: float = 0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Iterative pairwise repulsion. Returns (new_pos, converged)."""
    n_hard = benchmark.num_hard_macros
    if n_hard <= 1:
        return pos.clone(), True

    sizes = benchmark.macro_sizes[:n_hard].numpy()
    pos_np = pos[:n_hard].numpy().copy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + margin
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + margin

    converged = False
    for sweep in range(max_sweeps):
        any_overlap = False
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                if not movable[i] and not movable[j]:
                    continue
                dx = abs(pos_np[i, 0] - pos_np[j, 0])
                dy = abs(pos_np[i, 1] - pos_np[j, 1])
                sx = sep_x_mat[i, j]
                sy = sep_y_mat[i, j]
                if dx < sx and dy < sy:
                    pen_x = sx - dx
                    pen_y = sy - dy
                    any_overlap = True
                    if pen_x < pen_y:
                        shift = pen_x / 2 + 0.01
                        if pos_np[i, 0] <= pos_np[j, 0]:
                            if movable[i]:
                                pos_np[i, 0] -= shift
                            if movable[j]:
                                pos_np[j, 0] += shift
                        else:
                            if movable[i]:
                                pos_np[i, 0] += shift
                            if movable[j]:
                                pos_np[j, 0] -= shift
                    else:
                        shift = pen_y / 2 + 0.01
                        if pos_np[i, 1] <= pos_np[j, 1]:
                            if movable[i]:
                                pos_np[i, 1] -= shift
                            if movable[j]:
                                pos_np[j, 1] += shift
                        else:
                            if movable[i]:
                                pos_np[i, 1] += shift
                            if movable[j]:
                                pos_np[j, 1] -= shift
        pos_np[:, 0] = np.clip(pos_np[:, 0], half_w, cw - half_w)
        pos_np[:, 1] = np.clip(pos_np[:, 1], half_h, ch - half_h)
        if not any_overlap:
            converged = True
            if verbose:
                print(f"      [stage1] converged in {sweep + 1} sweeps")
            break

    if not converged and verbose:
        print(f"      [stage1] NOT converged in {max_sweeps} sweeps")

    result = pos.clone()
    result[:n_hard] = torch.tensor(pos_np, dtype=result.dtype)
    return result, converged


# ─── Stage 2: jiggle + retry ──────────────────────────────────────────────

def jiggle_retry(
    pos: torch.Tensor,
    benchmark,
    n_attempts: int = 5,
    jiggle_um: float = 0.5,
    seed: int = 42,
    max_sweeps: int = 200,
    margin: float = 0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool, int]:
    """Random perturbation + stage1 retry. Returns (new_pos, clean, attempts_used)."""
    n_hard = benchmark.num_hard_macros
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    rng = np.random.default_rng(seed)

    best = pos.clone()
    best_report = check_overlaps(best, benchmark, margin=margin)
    if best_report.n_overlaps == 0:
        return best, True, 0

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    for attempt in range(1, n_attempts + 1):
        offending = sorted({p.i for p in best_report.pairs} | {p.j for p in best_report.pairs})

        candidate = best.clone()
        cand_np = candidate[:n_hard].numpy().copy().astype(np.float64)
        amp = jiggle_um * attempt
        for idx in offending:
            if not movable[idx]:
                continue
            cand_np[idx, 0] += rng.uniform(-amp, amp)
            cand_np[idx, 1] += rng.uniform(-amp, amp)
        cand_np[:, 0] = np.clip(cand_np[:, 0], half_w, cw - half_w)
        cand_np[:, 1] = np.clip(cand_np[:, 1], half_h, ch - half_h)
        candidate[:n_hard] = torch.tensor(cand_np, dtype=candidate.dtype)

        candidate, _ = min_displacement_legalize(
            candidate, benchmark, max_sweeps=max_sweeps, margin=margin, verbose=False
        )
        report = check_overlaps(candidate, benchmark, margin=margin)
        if verbose:
            print(f"      [stage2] attempt {attempt}: amp={amp:.2f}μm "
                  f"jiggled {len(offending)} -> {report.n_overlaps} overlaps")

        if report.n_overlaps == 0:
            return candidate, True, attempt

        if (report.n_overlaps < best_report.n_overlaps or
                (report.n_overlaps == best_report.n_overlaps
                 and report.total_area < best_report.total_area)):
            best = candidate
            best_report = report

    return best, False, n_attempts


# ─── Stage 3: force-push (unconditional minimum-translation) ──────────────

def force_push(
    pos: torch.Tensor,
    benchmark,
    margin: float = 0.05,
    max_iterations: int = 500,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool, int]:
    """Greedy: always push largest-area overlapping pair apart on cheap axis.
    Returns (new_pos, clean, iterations_used).
    """
    n_hard = benchmark.num_hard_macros
    if n_hard <= 1:
        return pos.clone(), True, 0

    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    cur = pos.clone()
    iterations = 0
    for p in range(max_iterations):
        report = check_overlaps(cur, benchmark, margin=margin)
        if report.n_overlaps == 0:
            if verbose:
                print(f"      [stage3] cleaned in {p} iterations")
            return cur, True, p

        iterations = p + 1
        pairs_sorted = sorted(report.pairs, key=lambda x: -x.area)
        cur_np = cur[:n_hard].numpy().copy().astype(np.float64)
        for pair in pairs_sorted:
            i, j = pair.i, pair.j
            dx = abs(cur_np[i, 0] - cur_np[j, 0])
            dy = abs(cur_np[i, 1] - cur_np[j, 1])
            sep_x = (sizes[i, 0] + sizes[j, 0]) / 2 + margin
            sep_y = (sizes[i, 1] + sizes[j, 1]) / 2 + margin
            pen_x = sep_x - dx
            pen_y = sep_y - dy
            if pen_x <= 0 or pen_y <= 0:
                continue

            use_x = pen_x < pen_y
            shift = (pen_x if use_x else pen_y) + 0.02
            mov_i = movable[i]
            mov_j = movable[j]
            if not mov_i and not mov_j:
                continue
            axis = 0 if use_x else 1
            if mov_i and mov_j:
                if cur_np[i, axis] <= cur_np[j, axis]:
                    cur_np[i, axis] -= shift / 2
                    cur_np[j, axis] += shift / 2
                else:
                    cur_np[i, axis] += shift / 2
                    cur_np[j, axis] -= shift / 2
            elif mov_i:
                if cur_np[i, axis] <= cur_np[j, axis]:
                    cur_np[i, axis] -= shift
                else:
                    cur_np[i, axis] += shift
            else:
                if cur_np[j, axis] <= cur_np[i, axis]:
                    cur_np[j, axis] -= shift
                else:
                    cur_np[j, axis] += shift

        cur_np[:, 0] = np.clip(cur_np[:, 0], half_w, cw - half_w)
        cur_np[:, 1] = np.clip(cur_np[:, 1], half_h, ch - half_h)
        cur[:n_hard] = torch.tensor(cur_np, dtype=cur.dtype)

    final = check_overlaps(cur, benchmark, margin=margin)
    if verbose:
        print(f"      [stage3] after {max_iterations} iters: {final.n_overlaps} overlaps")
    return cur, final.n_overlaps == 0, iterations


# ─── Stage 4: nuclear fallback ────────────────────────────────────────────

def nuclear_fallback(
    benchmark,
    margin: float = 0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Restore CT initial.plc positions, then run a quick stage1 to clean
    any margin-related residuals.
    """
    placement = benchmark.macro_positions.clone().float()
    n_init = check_overlaps_torch(placement, benchmark, margin=margin)
    if n_init == 0:
        if verbose:
            print(f"      [stage4] CT positions clean")
        return placement, True

    if verbose:
        print(f"      [stage4] CT has {n_init} overlaps at margin={margin}; running stage1")
    cleaned, _ = min_displacement_legalize(
        placement, benchmark, max_sweeps=300, margin=margin, verbose=False
    )
    n_final = check_overlaps_torch(cleaned, benchmark, margin=margin)
    return cleaned, n_final == 0


# ─── Public entry point ───────────────────────────────────────────────────

def clean_overlaps(
    pos: torch.Tensor,
    benchmark,
    legal_max_sweeps: int = 300,
    jiggle_attempts: int = 5,
    force_max_iterations: int = 500,
    margin: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, CleanupResult]:
    """
    Run the 4-stage cleanup pipeline. Returns the first clean placement.

    Default margin is 0.0, matching the eval validator's zero-tolerance
    strict overlap test. Use a small positive margin (e.g. 0.01) for
    extra safety against numerical noise; use 0 to match validator exactly.
    """
    res = CleanupResult()
    res.n_overlaps_in = check_overlaps_torch(pos, benchmark, margin=margin)

    # Already clean — no work needed
    if res.n_overlaps_in == 0:
        res.stage_used = "already-clean"
        res.n_overlaps_out = 0
        return pos.clone(), res

    n_hard = benchmark.num_hard_macros
    candidates: List[Tuple[int, torch.Tensor, str]] = []

    # ── Stage 1 ──
    t = time.time()
    out1, _ = min_displacement_legalize(
        pos, benchmark, max_sweeps=legal_max_sweeps, margin=margin, verbose=verbose
    )
    res.timings_s["stage1"] = round(time.time() - t, 2)
    n1 = check_overlaps_torch(out1, benchmark, margin=margin)
    candidates.append((n1, out1, "stage1"))
    if n1 == 0:
        res.stage_used = "stage1"
        res.n_overlaps_out = 0
        _set_disp(res, pos, out1, benchmark)
        return out1, res

    if verbose:
        print(f"    [pipeline] stage1 left {n1} overlaps; trying stage2")

    # ── Stage 2 ──
    t = time.time()
    out2, _, attempts = jiggle_retry(
        out1, benchmark, n_attempts=jiggle_attempts,
        max_sweeps=legal_max_sweeps, margin=margin, verbose=verbose,
    )
    res.timings_s["stage2"] = round(time.time() - t, 2)
    res.notes.append(f"stage2_attempts={attempts}")
    n2 = check_overlaps_torch(out2, benchmark, margin=margin)
    candidates.append((n2, out2, "stage2"))
    if n2 == 0:
        res.stage_used = "stage2"
        res.n_overlaps_out = 0
        _set_disp(res, pos, out2, benchmark)
        return out2, res

    if verbose:
        print(f"    [pipeline] stage2 left {n2} overlaps; trying stage3")

    # ── Stage 3 ──
    t = time.time()
    out3, _, iters = force_push(
        out2, benchmark, margin=margin,
        max_iterations=force_max_iterations, verbose=verbose,
    )
    res.timings_s["stage3"] = round(time.time() - t, 2)
    res.notes.append(f"stage3_iters={iters}")
    n3 = check_overlaps_torch(out3, benchmark, margin=margin)
    candidates.append((n3, out3, "stage3"))
    if n3 == 0:
        res.stage_used = "stage3"
        res.n_overlaps_out = 0
        _set_disp(res, pos, out3, benchmark)
        return out3, res

    if verbose:
        print(f"    [pipeline] stage3 left {n3} overlaps; NUCLEAR FALLBACK")

    # ── Stage 4 (nuclear) ──
    t = time.time()
    out4, _ = nuclear_fallback(benchmark, margin=margin, verbose=verbose)
    res.timings_s["stage4"] = round(time.time() - t, 2)
    n4 = check_overlaps_torch(out4, benchmark, margin=margin)
    candidates.append((n4, out4, "stage4"))
    if n4 == 0:
        res.stage_used = "stage4"
        res.n_overlaps_out = 0
        res.notes.append("nuclear: returned CT positions")
        _set_disp(res, pos, out4, benchmark)
        return out4, res

    # ── Total failure: return best candidate ──
    best_n, best_pos, best_stage = min(candidates, key=lambda x: x[0])
    res.stage_used = "failed"
    res.n_overlaps_out = best_n
    res.notes.append(f"all stages failed; returning {best_stage} with {best_n} overlaps")
    _set_disp(res, pos, best_pos, benchmark)
    return best_pos, res


def _set_disp(res: CleanupResult, before: torch.Tensor, after: torch.Tensor, benchmark):
    n_hard = benchmark.num_hard_macros
    disp = (after[:n_hard] - before[:n_hard]).norm(dim=1)
    res.avg_displacement = float(disp.mean())
    res.max_displacement = float(disp.max())
