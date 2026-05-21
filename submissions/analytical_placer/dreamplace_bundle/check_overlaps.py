"""
Standalone overlap checker for hard macros.

Two public functions:

    check_overlaps(pos, benchmark, margin=0.05) -> OverlapReport
        Returns full structured report (pairs, total area, max penetration).

    check_overlaps_torch(pos, benchmark, margin=0.05) -> int
        Returns just the count of overlapping pairs. Used by the sweep
        for quick zero-overlap checks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch


@dataclass
class OverlapPair:
    i: int
    j: int
    pen_x: float        # x-axis penetration (positive = overlapping)
    pen_y: float        # y-axis penetration
    area: float         # overlap area = pen_x * pen_y
    min_pen: float      # min(pen_x, pen_y) — the cheap separation axis
    sep_axis: str       # "x" or "y"


@dataclass
class OverlapReport:
    pairs: List[OverlapPair] = field(default_factory=list)
    total_area: float = 0.0
    max_pen: float = 0.0

    @property
    def n_overlaps(self) -> int:
        return len(self.pairs)


def check_overlaps(pos: torch.Tensor, benchmark, margin: float = 0.05) -> OverlapReport:
    """Detect all hard-macro overlaps. Vectorized O(n²) numpy."""
    n_hard = benchmark.num_hard_macros
    if n_hard <= 1:
        return OverlapReport()

    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    pos_np = pos[:n_hard].numpy().astype(np.float64)

    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + margin
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + margin

    dx = np.abs(pos_np[:, 0:1] - pos_np[:, 0:1].T)
    dy = np.abs(pos_np[:, 1:2] - pos_np[:, 1:2].T)
    overlap = (dx < sep_x_mat) & (dy < sep_y_mat)
    overlap = np.triu(overlap, k=1)
    ii, jj = np.where(overlap)

    report = OverlapReport()
    for i, j in zip(ii.tolist(), jj.tolist()):
        pen_x = float(sep_x_mat[i, j] - dx[i, j])
        pen_y = float(sep_y_mat[i, j] - dy[i, j])
        if pen_x <= 0 or pen_y <= 0:
            continue
        area = pen_x * pen_y
        min_pen = min(pen_x, pen_y)
        report.pairs.append(OverlapPair(
            i=i, j=j,
            pen_x=pen_x, pen_y=pen_y,
            area=area, min_pen=min_pen,
            sep_axis="x" if pen_x < pen_y else "y",
        ))
        report.total_area += area
        if min_pen > report.max_pen:
            report.max_pen = min_pen

    return report


def check_overlaps_torch(pos: torch.Tensor, benchmark, margin: float = 0.05) -> int:
    """Quick path: just return the count of overlapping pairs."""
    n_hard = benchmark.num_hard_macros
    if n_hard <= 1:
        return 0

    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    pos_np = pos[:n_hard].numpy().astype(np.float64)

    sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + margin
    sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + margin

    dx = np.abs(pos_np[:, 0:1] - pos_np[:, 0:1].T)
    dy = np.abs(pos_np[:, 1:2] - pos_np[:, 1:2].T)
    overlap = (dx < sep_x_mat) & (dy < sep_y_mat)
    overlap = np.triu(overlap, k=1)
    return int(overlap.sum())


def format_report(report: OverlapReport, max_print: int = 10) -> str:
    """Pretty-print an OverlapReport for logs."""
    if report.n_overlaps == 0:
        return "  [check_overlaps] CLEAN — 0 overlaps"
    lines = [
        f"  [check_overlaps] {report.n_overlaps} overlaps, "
        f"total_area={report.total_area:.6f}μm², max_pen={report.max_pen:.4f}μm"
    ]
    for p in report.pairs[:max_print]:
        lines.append(
            f"    ({p.i:>4},{p.j:>4})  pen_x={p.pen_x:.4f} pen_y={p.pen_y:.4f}  "
            f"area={p.area:.6f}μm²  cheapest={p.sep_axis} ({p.min_pen:.4f}μm)"
        )
    if report.n_overlaps > max_print:
        lines.append(f"    ...{report.n_overlaps - max_print} more")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse, os, sys

    p = argparse.ArgumentParser()
    p.add_argument("benchmark_dir")
    p.add_argument("pl_path", nargs="?", default=None,
                   help="DREAMPlace .gp.pl output (omit to check CT initial)")
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from macro_place.loader import load_benchmark_from_dir
    from dreamplace_integration.pb2bookshelf import read_dreamplace_output

    benchmark, _ = load_benchmark_from_dir(args.benchmark_dir)
    if args.pl_path is None:
        placement = benchmark.macro_positions.float()
        print(f"Checking CT initial.plc for {benchmark.name}")
    else:
        positions = read_dreamplace_output(args.pl_path, args.benchmark_dir)
        placement = benchmark.macro_positions.clone().float()
        for i in range(benchmark.num_macros):
            nm = benchmark.macro_names[i].replace(" ", "_").replace("/", "_")
            if nm in positions:
                cx, cy = positions[nm]
                placement[i, 0] = cx
                placement[i, 1] = cy
        print(f"Checking {args.pl_path} for {benchmark.name}")

    rpt = check_overlaps(placement, benchmark)
    print(format_report(rpt))
