"""
Coordinate descent on the REAL TILOS proxy.

Uses the actual TILOS evaluator (not an approximation) to guide macro moves.
Every accepted move is guaranteed to improve the real score.

This is the key post-processing step that reduces congestion by moving
high-degree macros away from routing hotspots.
"""

from __future__ import annotations
import numpy as np
import torch
import time
from typing import Optional, Tuple

from macro_place.objective import compute_proxy_cost
from macro_place.objective import _set_placement as set_placement_in_plc


def abu5_shift(
    placement: torch.Tensor,
    benchmark,
    plc,
    max_iters: int = 10,
    top_k_macros: int = 10,
    step_grid_cells: float = 1.0,
    time_budget_s: float = 300.0,
    verbose: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Coordinate descent on the REAL TILOS proxy.

    For each iteration:
    1. Rank macros by net degree (high-degree = most impact on congestion)
    2. For top-K macros, try 8 directional moves
    3. Accept the single best move that reduces proxy
    4. Repeat until no improvement or budget exhausted
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_w = cw / plc.grid_col
    grid_h = ch / plc.grid_row
    step_x = step_grid_cells * grid_w
    step_y = step_grid_cells * grid_h

    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    sizes = benchmark.macro_sizes[:n_hard].float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    best_pos = placement.clone()
    set_placement_in_plc(plc, best_pos, benchmark)
    best_proxy = float(compute_proxy_cost(best_pos, benchmark, plc)["proxy_cost"])
    initial_proxy = best_proxy

    info = {
        "initial_proxy": initial_proxy,
        "final_proxy": best_proxy,
        "accepted_moves": 0,
        "iterations": 0,
        "time_s": 0.0,
        "improvement": 0.0,
    }

    if verbose:
        print(f"[abu5_shift] starting proxy={initial_proxy:.4f}")

    t_start = time.time()

    # Rank macros by net degree
    net_nodes = benchmark.net_nodes
    macro_degree = np.zeros(n_hard)
    for i in range(benchmark.num_nets):
        for node_idx in net_nodes[i].tolist():
            if node_idx < n_hard:
                macro_degree[node_idx] += 1

    candidates = []
    for idx in np.argsort(-macro_degree):
        if movable[idx]:
            candidates.append(int(idx))
        if len(candidates) >= top_k_macros:
            break

    directions = [(step_x, 0), (-step_x, 0), (0, step_y), (0, -step_y),
                  (step_x * 0.7, step_y * 0.7), (-step_x * 0.7, step_y * 0.7),
                  (step_x * 0.7, -step_y * 0.7), (-step_x * 0.7, -step_y * 0.7)]

    for iteration in range(max_iters):
        if time.time() - t_start > time_budget_s:
            break

        improved = False
        pos_np = best_pos[:n_hard].numpy().copy()

        for macro_idx in candidates:
            if time.time() - t_start > time_budget_s:
                break

            cx = float(pos_np[macro_idx, 0])
            cy = float(pos_np[macro_idx, 1])
            best_move_proxy = best_proxy
            best_move_pos = None

            for dx, dy in directions:
                new_cx = max(float(half_w[macro_idx]),
                             min(cw - float(half_w[macro_idx]), cx + dx))
                new_cy = max(float(half_h[macro_idx]),
                             min(ch - float(half_h[macro_idx]), cy + dy))

                # Quick overlap check
                has_overlap = False
                for j in range(n_hard):
                    if j == macro_idx:
                        continue
                    sep_x = (float(sizes[macro_idx, 0]) + float(sizes[j, 0])) / 2
                    sep_y = (float(sizes[macro_idx, 1]) + float(sizes[j, 1])) / 2
                    if (abs(new_cx - pos_np[j, 0]) < sep_x and
                            abs(new_cy - pos_np[j, 1]) < sep_y):
                        has_overlap = True
                        break
                if has_overlap:
                    continue

                # Evaluate REAL proxy
                trial_pos = best_pos.clone()
                trial_pos[macro_idx, 0] = new_cx
                trial_pos[macro_idx, 1] = new_cy
                set_placement_in_plc(plc, trial_pos, benchmark)
                trial_proxy = float(
                    compute_proxy_cost(trial_pos, benchmark, plc)["proxy_cost"]
                )

                if trial_proxy < best_move_proxy:
                    best_move_proxy = trial_proxy
                    best_move_pos = trial_pos.clone()

            if best_move_pos is not None and best_move_proxy < best_proxy:
                best_pos = best_move_pos
                best_proxy = best_move_proxy
                pos_np = best_pos[:n_hard].numpy().copy()
                info["accepted_moves"] += 1
                improved = True

        if not improved:
            step_x *= 0.6
            step_y *= 0.6
            directions = [(step_x, 0), (-step_x, 0), (0, step_y), (0, -step_y),
                          (step_x * 0.7, step_y * 0.7), (-step_x * 0.7, step_y * 0.7),
                          (step_x * 0.7, -step_y * 0.7), (-step_x * 0.7, -step_y * 0.7)]
            if step_x < grid_w * 0.2:
                break

        info["iterations"] = iteration + 1

    info["final_proxy"] = best_proxy
    info["time_s"] = round(time.time() - t_start, 1)
    info["improvement"] = initial_proxy - best_proxy

    if verbose:
        print(f"[abu5_shift] done: {initial_proxy:.4f} → {best_proxy:.4f} "
              f"(saved {info['improvement']:.4f}, "
              f"{info['accepted_moves']} moves, {info['time_s']}s)")

    return best_pos, info
