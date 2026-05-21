"""
v5 Sweep — uses placer.py's DreamplacePlacer directly so we get the
multi-mode adaptive strategy (default + congestion_aware, plus fine_grid
+ default_seed2 on high-proxy benchmarks).

Per-benchmark output mirrors the v3/v4 sweep output format so we can
compare directly.

Usage (in WSL2 Ubuntu shell):
    cd /mnt/c/Users/iamkr/projects/Hackathons/hrt2/macro-place-challenge-2026
    python3 dreamplace_integration/sweep_v5.py
"""

import os
import sys
import time
import json
import traceback
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
from dreamplace_integration.check_overlaps import check_overlaps_torch


ICCAD04_DIR = os.path.join(_ROOT, "external", "MacroPlacement", "Testcases", "ICCAD04")
DEFAULT_AVG_CURRENT = 1.4738   # the legacy 1.4738 placer
DEFAULT_AVG_V4 = 1.3156        # v4 locked baseline


def load_placer():
    """Load DreamplacePlacer from submissions/analytical_placer/placer.py."""
    placer_path = os.path.join(
        _ROOT, "submissions", "analytical_placer", "placer.py"
    )
    spec = importlib.util.spec_from_file_location("placer_v5", placer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DreamplacePlacer()


def evaluate_one(placer, name: str, bench_dir: str) -> dict:
    """Run placer on one benchmark and validate the result."""
    result = {"name": name, "ok": False}
    t0 = time.time()
    try:
        benchmark, plc = load_benchmark_from_dir(bench_dir)
        placement = placer.place(benchmark)
    except Exception as e:
        result["error"] = str(e)
        result["trace"] = traceback.format_exc()
        return result

    is_valid, violations = validate_placement(placement, benchmark)
    n_strict = check_overlaps_torch(placement, benchmark, margin=0.0)
    costs = compute_proxy_cost(placement, benchmark, plc)
    ct_costs = compute_proxy_cost(benchmark.macro_positions.float(), benchmark, plc)

    result.update({
        "ok": True,
        "proxy": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "ct_proxy": float(ct_costs["proxy_cost"]),
        "ct_density": float(ct_costs["density_cost"]),
        "ct_congestion": float(ct_costs["congestion_cost"]),
        "valid": is_valid,
        "n_violations": len(violations),
        "n_strict_overlaps": n_strict,
        "metric_overlaps": costs["overlap_count"],
        "n_macros": benchmark.num_macros,
        "n_hard_macros": benchmark.num_hard_macros,
        "runtime_s": round(time.time() - t0, 1),
    })
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these benchmarks (e.g. ibm06 ibm15)")
    p.add_argument("--out", default=None,
                   help="output json path (default: dreamplace_integration/sweep_results_v5.json)")
    args = p.parse_args()

    benchmarks = sorted(
        d for d in os.listdir(ICCAD04_DIR)
        if d.startswith("ibm") and os.path.isdir(os.path.join(ICCAD04_DIR, d))
    )
    if args.only:
        benchmarks = [b for b in benchmarks if b in args.only]

    print(f"v5 multi-mode sweep — {len(benchmarks)} benchmarks: {benchmarks}")
    print(f"Loading placer...", flush=True)
    placer = load_placer()
    print(f"Modes: DEFAULT={placer.DEFAULT_MODES}  EXTRA={placer.EXTRA_MODES} "
          f"(threshold={placer.EXTRA_MODE_THRESHOLD})\n", flush=True)

    out_json = args.out or os.path.join(
        _ROOT, "dreamplace_integration", "sweep_results_v5.json"
    )

    results = []
    t_sweep = time.time()
    for name in benchmarks:
        bench_dir = os.path.join(ICCAD04_DIR, name)
        print(f"\n{'='*70}\n[{name}] starting\n{'='*70}", flush=True)
        r = evaluate_one(placer, name, bench_dir)
        results.append(r)

        if r["ok"]:
            print(f"\n[{name}] proxy={r['proxy']:.4f} (CT={r['ct_proxy']:.4f}) "
                  f"wl={r['wirelength']:.4f} den={r['density']:.4f} "
                  f"cong={r['congestion']:.4f} "
                  f"valid={r['valid']} ovlp={r['n_strict_overlaps']} "
                  f"runtime={r['runtime_s']}s", flush=True)
        else:
            print(f"[{name}] FAILED: {r.get('error', 'unknown')}", flush=True)

        # Save partial after each benchmark
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    sweep_runtime = time.time() - t_sweep
    print(f"\n\n{'='*100}")
    print("v5 SWEEP SUMMARY")
    print("=" * 100)
    fmt = "{:>8} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>5} | {:>6}"
    print(fmt.format(
        "bench", "v5", "v4_ref", "delta", "wl", "den", "cong", "ovlp", "time_s"
    ))
    print("-" * 100)
    ok = [r for r in results if r.get("ok")]
    for r in ok:
        v4 = _v4_baseline_proxy(r["name"])
        delta_str = "—"
        if v4 is not None:
            d = r["proxy"] - v4
            delta_str = f"{'+' if d > 0 else ''}{d:.4f}"
        print(fmt.format(
            r["name"],
            f"{r['proxy']:.4f}",
            f"{v4:.4f}" if v4 is not None else "—",
            delta_str,
            f"{r['wirelength']:.4f}",
            f"{r['density']:.3f}",
            f"{r['congestion']:.3f}",
            r["n_strict_overlaps"],
            f"{r['runtime_s']:.0f}",
        ))

    if ok:
        avg_v5 = sum(r["proxy"] for r in ok) / len(ok)
        print("-" * 100)
        print(f"AVG v5 = {avg_v5:.4f}")
        print(f"v4 baseline = {DEFAULT_AVG_V4}  → v5 vs v4: "
              f"{(DEFAULT_AVG_V4 - avg_v5) / DEFAULT_AVG_V4 * 100:+.2f}%")
        print(f"1.4738 baseline → v5: "
              f"{(DEFAULT_AVG_CURRENT - avg_v5) / DEFAULT_AVG_CURRENT * 100:+.2f}%")
        clean = sum(1 for r in ok if r["n_strict_overlaps"] == 0)
        print(f"Validity: {clean}/{len(ok)} benchmarks have 0 strict overlaps")

    failed = [r for r in results if not r.get("ok")]
    if failed:
        print(f"\nFAILED: {[r['name'] for r in failed]}")
    print(f"\nTotal sweep wall-clock: {sweep_runtime:.0f}s ({sweep_runtime/60:.1f} min)")
    print(f"Results: {out_json}")


# ── v4 baselines for delta reporting ──────────────────────────────────────────
_V4_BASELINES = {
    "ibm01": 0.9287, "ibm02": 1.3335, "ibm03": 1.1113, "ibm04": 1.2159,
    "ibm06": 1.6069, "ibm07": 1.3560, "ibm08": 1.4498, "ibm09": 0.9850,
    "ibm10": 1.2467, "ibm11": 1.0524, "ibm12": 1.4095, "ibm13": 1.1204,
    "ibm14": 1.3944, "ibm15": 1.5338, "ibm16": 1.3650, "ibm17": 1.5931,
    "ibm18": 1.6630,
}


def _v4_baseline_proxy(name: str):
    return _V4_BASELINES.get(name)


if __name__ == "__main__":
    main()
