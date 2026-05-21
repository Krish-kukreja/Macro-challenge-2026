"""
DREAMPlace-based macro placer with safety fallback.

Pipeline per benchmark:
  1. Convert TILOS .pb.txt -> Bookshelf format (in-memory writer)
  2. Subprocess-call the BUNDLED DREAMPlace Placer.py
  3. Read output .gp.pl, map back to TILOS macro indices
  4. Run 4-stage cleanup (min-disp legalize / jiggle / force-push / nuclear)
  5. If anything failed, fall back to the 1.4738-quality placer (placer_backup_1.4738.py)

Submission layout expected at runtime:
    /submission/placer_dreamplace.py        <- this file
    /submission/dreamplace_bundle/          <- the bundled DREAMPlace install
        dreamplace/                         (package, ~73MB)
        lib/libcudart.so.11.0               (CUDA 11.8 runtime, ~700KB)
        _bundle_env.py                      (env helper)
    /submission/placer_backup_1.4738.py     <- safety fallback

The bundle directory must be sibling to placer_dreamplace.py.
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# Resolve bundle paths relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.join(_HERE, "dreamplace_bundle")


def _bundle_available() -> bool:
    """Check whether the DREAMPlace bundle is present and usable."""
    return (
        os.path.isdir(_BUNDLE_DIR)
        and os.path.isfile(os.path.join(_BUNDLE_DIR, "dreamplace", "Placer.py"))
        and os.path.isfile(os.path.join(_BUNDLE_DIR, "_bundle_env.py"))
    )


def _import_bundle_env():
    """Import _bundle_env from the bundle directory."""
    if _BUNDLE_DIR not in sys.path:
        sys.path.insert(0, _BUNDLE_DIR)
    import _bundle_env  # noqa: E402
    return _bundle_env


def _get_pb2bookshelf():
    """Locate the pb2bookshelf converter regardless of how the placer was loaded."""
    # Prefer bundled copy (self-contained for eval Docker)
    if _BUNDLE_DIR not in sys.path:
        sys.path.insert(0, _BUNDLE_DIR)
    try:
        from pb2bookshelf import convert, read_dreamplace_output  # type: ignore
        return convert, read_dreamplace_output
    except ImportError:
        pass
    # Fallback to repo layout (local dev)
    try:
        from dreamplace_integration.pb2bookshelf import convert, read_dreamplace_output
        return convert, read_dreamplace_output
    except ImportError:
        pass
    raise ImportError("Cannot locate pb2bookshelf in bundle or dreamplace_integration")


def _get_cleanup_pipeline():
    """Locate the post-legalize pipeline."""
    if _BUNDLE_DIR not in sys.path:
        sys.path.insert(0, _BUNDLE_DIR)
    try:
        from post_legalize import clean_overlaps  # type: ignore
        return clean_overlaps
    except ImportError:
        pass
    try:
        from dreamplace_integration.post_legalize import clean_overlaps
        return clean_overlaps
    except ImportError:
        pass
    raise ImportError("Cannot locate post_legalize in bundle or dreamplace_integration")


def _fallback_placer_class():
    """Load the 1.4738 backup placer as a safety fallback."""
    backup_path = os.path.join(_HERE, "placer_backup_1.4738.py")
    if not os.path.isfile(backup_path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("placer_backup_1_4738", backup_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Find the placer class (mirrors macro_place.evaluate's loader convention)
    for attr in vars(mod).values():
        if (isinstance(attr, type)
                and callable(getattr(attr, "place", None))
                and attr.__module__ == "placer_backup_1_4738"):
            return attr
    return None


class DreamplacePlacer:
    """
    Macro placer using bundled DREAMPlace + 4-stage overlap cleanup.

    Convention required by macro_place.evaluate: a `place(benchmark)` method
    that returns a [num_macros, 2] torch.Tensor.
    """

    # Per-benchmark timeout (seconds) for DREAMPlace global placement.
    # Must be under 3600s (1 hour) per competition rules.
    # Set to 3000s to leave margin for cleanup + checkpoint scan.
    DP_TIMEOUT_S = 600  # per-MODE timeout (each mode gets 10 min max)

    # Total per-benchmark wall-clock budget (hard cap)
    BENCHMARK_TIMEOUT_S = 3000  # 50 min total — leaves 10 min margin

    # Best-proxy checkpoint selection — set USE_CHECKPOINTS=False to disable.
    USE_CHECKPOINTS = True
    CHECKPOINT_EVERY = 50              # save positions every N iterations
    MIN_CHECKPOINT_ITER = 300          # skip pre-convergence iters when scanning
    CHECKPOINT_SCAN_STRIDE = 2         # scan every Nth checkpoint to bound cost
    CHECKPOINT_SCAN_TIME_S = 60.0      # wall-clock cap for ckpt scan per benchmark

    # Multi-mode DREAMPlace strategy. We run multiple configurations and
    # pick the one that yields the best clean proxy.
    #
    # Modes (definitions in pb2bookshelf.py):
    #   default          — vanilla v4 settings (512 bins, density_weight 8e-5)
    #   congestion_aware — tighter density convergence + 2× density_weight
    #                      (no NCTUgr binary needed)
    #   fine_grid        — 1024 bins + 3000 iters, tighter overflow stop
    #   default_seed2    — same as default but random_seed=2000 (variance reduction)
    #
    # Strategy: every benchmark runs DEFAULT_MODES (cheap, ~2x time of v4).
    # Benchmarks where the default proxy is HIGH (>=1.45) ALSO run EXTRA_MODES
    # (more expensive but worth it because the proxy gap is bigger to cover).
    DEFAULT_MODES = ["default", "congestion_aware"]
    EXTRA_MODES = ["fine_grid", "default_seed2", "default_seed3", "gift_init"]
    EXTRA_MODE_THRESHOLD = 1.45        # only add extra modes when default ≥ this
    MODES = DEFAULT_MODES              # back-compat: single static list
    USE_ADAPTIVE_MODES = True          # if True, use the threshold logic

    # ABU5-targeted congestion shifting (post-processing after cleanup)
    # Bug fixed: argument order corrected in abu5_shifter.py
    USE_ABU5_SHIFT = True
    ABU5_MAX_ITERS = 20
    ABU5_TIME_BUDGET_S = 60.0         # reduced to fit in benchmark timeout

    # Differentiable proxy fine-tuning (runs after DREAMPlace + cleanup)
    USE_DIFF_OPT = True
    DIFF_OPT_STEPS = 5000             # more steps for better convergence
    DIFF_OPT_TIME_S = 300.0           # 5 min budget per pass
    DIFF_OPT_TWO_PASS = True          # coarse then fine
    DIFF_OPT_COARSE_LR = 0.005
    DIFF_OPT_FINE_LR = 0.0008         # even finer for polishing
    DIFF_OPT_COARSE_TIME_S = 200.0    # more time for coarse pass
    DIFF_OPT_FINE_TIME_S = 400.0      # more time for fine pass
    DIFF_OPT_ITERATIVE = True         # run 3 rounds of opt+cleanup
    DIFF_OPT_ROUNDS = 3

    # Per-benchmark config overrides — REMOVED to comply with "no hardcoding
    # solutions for specific benchmarks" rule. Instead, multi-seed is applied
    # uniformly via EXTRA_MODES to all high-proxy benchmarks.
    BENCHMARK_OVERRIDE = {}  # empty — general algorithm only

    # Macro orientation search (Klein-4: N, FN, FS, S)
    USE_ORIENTATION_SEARCH = True
    ORIENTATION_TOP_K = 20            # only try top-K macros by net degree

    # Multi-start: also try diff_opt from multiple random grid initializations
    USE_MULTI_START_RANDOM = True
    N_RANDOM_STARTS = 3               # number of random grid starts to try

    # Per-mode timeout overrides (seconds).
    MODE_TIMEOUTS = {
        "default": 600,
        "default_seed2": 600,
        "default_seed3": 600,
        "congestion_aware": 600,
        "fine_grid": 900,
        "gift_init": 600,
        "override": 600,
    }

    def __init__(self):
        self._bundle_env = None
        self._fallback = None
        if _bundle_available():
            try:
                self._bundle_env = _import_bundle_env()
            except Exception as e:
                print(f"[placer_dreamplace] WARNING: bundle present but env helper "
                      f"failed: {e}", flush=True)

        # Always try to load the fallback so we have a safety net
        cls = _fallback_placer_class()
        self._fallback = cls() if cls else None

    # ── Public interface used by the eval harness ─────────────────────────

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Run DREAMPlace + cleanup. Falls back to the 1.4738 baseline placer
        if anything goes wrong or if benchmark timeout is exceeded.
        """
        self._benchmark_start = time.time()

        if self._bundle_env is None:
            print(f"[placer_dreamplace] no DREAMPlace bundle found; using fallback")
            return self._fallback_place(benchmark)

        try:
            placement = self._dreamplace_path(benchmark)
            runtime = time.time() - self._benchmark_start
            print(f"[placer_dreamplace] {benchmark.name} OK in "
                  f"{runtime:.1f}s", flush=True)
            return placement
        except Exception as e:
            print(f"[placer_dreamplace] {benchmark.name} FAILED: {e}; using fallback",
                  flush=True)
            return self._fallback_place(benchmark)

    # ── DREAMPlace pipeline ───────────────────────────────────────────────

    def _dreamplace_path(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Run DEFAULT_MODES first, pick the best. If best proxy >= EXTRA_MODE_THRESHOLD,
        also try EXTRA_MODES and use whichever wins overall.
        If benchmark has a BENCHMARK_OVERRIDE, also try that config.
        """
        bench_dir = self._infer_benchmark_dir(benchmark)
        if bench_dir is None or not os.path.isdir(bench_dir):
            raise RuntimeError(
                f"cannot locate source dir for {benchmark.name} "
                f"(tried {bench_dir})"
            )

        # Phase 1: cheap modes (run on every benchmark)
        modes_to_try = list(self.DEFAULT_MODES) if self.USE_ADAPTIVE_MODES else list(self.MODES)

        # If benchmark has a per-benchmark override, add it
        if benchmark.name in self.BENCHMARK_OVERRIDE:
            modes_to_try.append("override")

        best_clean, best_proxy, best_label, first_error = self._run_modes(
            benchmark, bench_dir, modes_to_try,
        )

        # Phase 2: if we're still high-proxy, try the expensive modes too
        if (self.USE_ADAPTIVE_MODES
                and best_proxy is not None
                and best_proxy >= self.EXTRA_MODE_THRESHOLD):
            print(f"[placer_dreamplace] {benchmark.name}: best so far "
                  f"{best_proxy:.4f} ≥ {self.EXTRA_MODE_THRESHOLD}, "
                  f"trying extra modes {self.EXTRA_MODES}", flush=True)
            extra_clean, extra_proxy, extra_label, extra_err = self._run_modes(
                benchmark, bench_dir, self.EXTRA_MODES,
            )
            if extra_proxy is not None and extra_proxy < best_proxy:
                best_clean = extra_clean
                best_proxy = extra_proxy
                best_label = extra_label

        if best_clean is None:
            raise RuntimeError(
                f"all modes failed; first error: {first_error}"
            )

        # Phase 3: Also try diff_opt directly from CT initial.plc (multi-start)
        # Different starting point may find a different (better) local minimum
        elapsed = time.time() - self._benchmark_start
        if self.USE_DIFF_OPT and elapsed < self.BENCHMARK_TIMEOUT_S - 300:
            try:
                from dreamplace_integration.diff_proxy_optimizer import optimize_proxy
                from macro_place.objective import compute_proxy_cost as _cpc

                ct_pos = benchmark.macro_positions.clone().float()
                ct_opt, _ = optimize_proxy(
                    ct_pos, benchmark, self._get_plc(bench_dir),
                    n_steps=self.DIFF_OPT_STEPS,
                    lr=self.DIFF_OPT_COARSE_LR,
                    time_budget_s=min(300.0, self.BENCHMARK_TIMEOUT_S - elapsed - 60),
                    verbose=False,
                )
                from dreamplace_integration.post_legalize import clean_overlaps
                ct_clean, ct_info = clean_overlaps(
                    ct_opt, benchmark,
                    legal_max_sweeps=300, jiggle_attempts=5,
                    force_max_iterations=500, margin=0.0, verbose=False,
                )
                if ct_info.n_overlaps_out == 0:
                    ct_proxy = float(_cpc(ct_clean, benchmark,
                                         self._get_plc(bench_dir))["proxy_cost"])
                    if ct_proxy < best_proxy:
                        best_clean = ct_clean
                        best_proxy = ct_proxy
                        best_label = "ct_initial+diffopt"
                        print(f"[placer_dreamplace] {benchmark.name}: "
                              f"CT+diffopt wins: {ct_proxy:.4f}", flush=True)
            except Exception as e:
                print(f"[placer_dreamplace] {benchmark.name}: "
                      f"CT+diffopt failed: {e}", flush=True)

        # Phase 4: Multi-start from random grid initializations
        if self.USE_MULTI_START_RANDOM and self.USE_DIFF_OPT:
            elapsed = time.time() - self._benchmark_start
            if elapsed < self.BENCHMARK_TIMEOUT_S - 400:
                try:
                    from dreamplace_integration.diff_proxy_optimizer import optimize_proxy
                    from macro_place.objective import compute_proxy_cost as _cpc
                    from dreamplace_integration.post_legalize import clean_overlaps
                    import torch

                    for start_i in range(self.N_RANDOM_STARTS):
                        elapsed = time.time() - self._benchmark_start
                        if elapsed > self.BENCHMARK_TIMEOUT_S - 200:
                            break
                        # Grid-based random initialization
                        n_mac = benchmark.num_macros
                        cw = float(benchmark.canvas_width)
                        ch = float(benchmark.canvas_height)
                        rng = torch.Generator().manual_seed(7777 + start_i * 1000)
                        rand_pos = benchmark.macro_positions.clone().float()
                        # Jitter positions by up to 30% of canvas
                        jitter_x = (torch.rand(n_mac, generator=rng) - 0.5) * cw * 0.3
                        jitter_y = (torch.rand(n_mac, generator=rng) - 0.5) * ch * 0.3
                        rand_pos[:, 0] += jitter_x
                        rand_pos[:, 1] += jitter_y
                        # Clamp
                        sizes = benchmark.macro_sizes[:n_mac].float()
                        rand_pos[:, 0] = rand_pos[:, 0].clamp(sizes[:, 0]/2, cw - sizes[:, 0]/2)
                        rand_pos[:, 1] = rand_pos[:, 1].clamp(sizes[:, 1]/2, ch - sizes[:, 1]/2)

                        rand_opt, _ = optimize_proxy(
                            rand_pos, benchmark, self._get_plc(bench_dir),
                            n_steps=3000, lr=0.005,
                            time_budget_s=min(150.0, self.BENCHMARK_TIMEOUT_S - elapsed - 60),
                            verbose=False,
                        )
                        rand_clean, rand_info = clean_overlaps(
                            rand_opt, benchmark,
                            legal_max_sweeps=300, jiggle_attempts=5,
                            force_max_iterations=500, margin=0.0, verbose=False,
                        )
                        if rand_info.n_overlaps_out == 0:
                            rand_proxy = float(_cpc(rand_clean, benchmark,
                                                    self._get_plc(bench_dir))["proxy_cost"])
                            if rand_proxy < best_proxy:
                                best_clean = rand_clean
                                best_proxy = rand_proxy
                                best_label = f"random_start_{start_i}+diffopt"
                                print(f"[placer_dreamplace] {benchmark.name}: "
                                      f"random start {start_i} wins: {rand_proxy:.4f}",
                                      flush=True)
                except Exception as e:
                    print(f"[placer_dreamplace] {benchmark.name}: "
                          f"random starts failed: {e}", flush=True)

        # Phase 5: Orientation search (try flipping top-K macros)
        if self.USE_ORIENTATION_SEARCH:
            elapsed = time.time() - self._benchmark_start
            if elapsed < self.BENCHMARK_TIMEOUT_S - 60:
                try:
                    best_clean, best_proxy = self._orientation_search(
                        best_clean, benchmark, bench_dir, best_proxy
                    )
                except Exception as e:
                    print(f"[placer_dreamplace] {benchmark.name}: "
                          f"orientation search failed: {e}", flush=True)

        print(f"[placer_dreamplace] {benchmark.name} best: "
              f"proxy={best_proxy:.4f} via {best_label}", flush=True)
        return best_clean

    def _run_modes(self, benchmark, bench_dir, modes):
        """Run a list of modes; return (best_clean, best_proxy, best_label, first_error)."""
        best_clean = None
        best_proxy = float("inf")
        best_label = None
        first_error = None
        for mode in modes:
            # Check benchmark-level timeout
            elapsed = time.time() - self._benchmark_start
            if elapsed > self.BENCHMARK_TIMEOUT_S:
                print(f"[placer_dreamplace] {benchmark.name}: "
                      f"benchmark timeout ({elapsed:.0f}s > {self.BENCHMARK_TIMEOUT_S}s), "
                      f"stopping modes", flush=True)
                break
            try:
                cand_clean, cand_proxy, cand_label = self._run_one_mode(
                    benchmark, bench_dir, mode,
                )
                print(f"[placer_dreamplace] {benchmark.name} mode={mode}: "
                      f"proxy={cand_proxy:.4f} ({cand_label})", flush=True)
                if cand_proxy < best_proxy:
                    best_clean = cand_clean
                    best_proxy = cand_proxy
                    best_label = f"{mode}/{cand_label}"
            except Exception as e:
                msg = f"mode={mode} failed: {e}"
                print(f"[placer_dreamplace] {benchmark.name} {msg}", flush=True)
                if first_error is None:
                    first_error = msg
        proxy_out = None if best_clean is None else best_proxy
        return best_clean, proxy_out, best_label, first_error

    def _run_one_mode(self, benchmark: Benchmark, bench_dir: str, mode: str):
        """Run a single DREAMPlace configuration; return (clean_pos, proxy, label).
        label is 'final' or 'ckpt-{iter}' to indicate where the placement came from.
        """
        convert, read_dreamplace_output = _get_pb2bookshelf()
        clean_overlaps = _get_cleanup_pipeline()
        timeout_s = self.MODE_TIMEOUTS.get(mode, self.DP_TIMEOUT_S)

        with tempfile.TemporaryDirectory(prefix=f"dp_{mode}_", dir="/tmp") as work_dir:
            json_path = convert(bench_dir, work_dir, gpu=1, mode=mode)
            log_path = os.path.join(work_dir, "dp.log")
            ckpt_dir = os.path.join(work_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            env = self._bundle_env.subprocess_env()
            if self.USE_CHECKPOINTS:
                env["DREAMPLACE_CHECKPOINT_DIR"] = ckpt_dir
                env["DREAMPLACE_CHECKPOINT_EVERY"] = str(self.CHECKPOINT_EVERY)

            with open(log_path, "w") as f:
                proc = subprocess.run(
                    [sys.executable, self._bundle_env.placer_py_path(), json_path],
                    cwd=self._bundle_env.bundle_root(),
                    env=env,
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                )
            if proc.returncode != 0:
                with open(log_path) as f:
                    tail = "\n".join(f.readlines()[-15:])
                raise RuntimeError(f"rc={proc.returncode}; tail:\n{tail}")

            pl_out = os.path.join(
                work_dir, "results", benchmark.name, f"{benchmark.name}.gp.pl"
            )
            if not os.path.isfile(pl_out):
                raise RuntimeError(f"no output .pl at {pl_out}")

            positions = read_dreamplace_output(pl_out, bench_dir)
            final_placement = self._positions_dict_to_tensor(positions, benchmark)
            final_clean, final_info = clean_overlaps(
                final_placement, benchmark,
                legal_max_sweeps=300, jiggle_attempts=5,
                force_max_iterations=500, margin=0.0, verbose=False,
            )
            if final_info.n_overlaps_out > 0:
                raise RuntimeError(
                    f"final cleanup left {final_info.n_overlaps_out} overlaps"
                )
            final_proxy = self._proxy(final_clean, bench_dir)
            best_clean = final_clean
            best_proxy = final_proxy
            best_label = "final"

            if self.USE_CHECKPOINTS:
                try:
                    cand_clean, cand_proxy, cand_iter = self._scan_checkpoints(
                        ckpt_dir, benchmark, bench_dir, clean_overlaps,
                    )
                    if cand_proxy is not None and cand_proxy < best_proxy:
                        best_clean = cand_clean
                        best_proxy = cand_proxy
                        best_label = f"ckpt-{cand_iter}"
                except Exception as e:
                    print(f"  [{mode}] ckpt scan failed: {e}", flush=True)

            # Step 6: Differentiable proxy fine-tuning (iterative rounds)
            if self.USE_DIFF_OPT:
                try:
                    from dreamplace_integration.diff_proxy_optimizer import optimize_proxy
                    from macro_place.objective import compute_proxy_cost as _cpc

                    current_best = best_clean
                    current_proxy = best_proxy
                    n_rounds = self.DIFF_OPT_ROUNDS if self.DIFF_OPT_ITERATIVE else 1

                    for round_i in range(n_rounds):
                        # Check benchmark timeout
                        elapsed = time.time() - self._benchmark_start
                        if elapsed > self.BENCHMARK_TIMEOUT_S - 120:
                            break

                        # Coarse pass
                        opt_pos, _ = optimize_proxy(
                            current_best, benchmark, self._get_plc(bench_dir),
                            n_steps=self.DIFF_OPT_STEPS,
                            lr=self.DIFF_OPT_COARSE_LR,
                            time_budget_s=self.DIFF_OPT_COARSE_TIME_S / n_rounds,
                            verbose=False,
                        )
                        opt_clean, opt_info = clean_overlaps(
                            opt_pos, benchmark,
                            legal_max_sweeps=300, jiggle_attempts=5,
                            force_max_iterations=500, margin=0.0, verbose=False,
                        )
                        if opt_info.n_overlaps_out == 0:
                            p = float(_cpc(opt_clean, benchmark,
                                           self._get_plc(bench_dir))["proxy_cost"])
                            if p < current_proxy:
                                current_best = opt_clean
                                current_proxy = p

                        # Fine pass (if two-pass enabled and time allows)
                        if self.DIFF_OPT_TWO_PASS:
                            elapsed = time.time() - self._benchmark_start
                            if elapsed > self.BENCHMARK_TIMEOUT_S - 120:
                                break
                            opt_pos2, _ = optimize_proxy(
                                current_best, benchmark, self._get_plc(bench_dir),
                                n_steps=self.DIFF_OPT_STEPS,
                                lr=self.DIFF_OPT_FINE_LR,
                                time_budget_s=self.DIFF_OPT_FINE_TIME_S / n_rounds,
                                verbose=False,
                            )
                            opt_clean2, opt_info2 = clean_overlaps(
                                opt_pos2, benchmark,
                                legal_max_sweeps=300, jiggle_attempts=5,
                                force_max_iterations=500, margin=0.0, verbose=False,
                            )
                            if opt_info2.n_overlaps_out == 0:
                                p2 = float(_cpc(opt_clean2, benchmark,
                                               self._get_plc(bench_dir))["proxy_cost"])
                                if p2 < current_proxy:
                                    current_best = opt_clean2
                                    current_proxy = p2

                    if current_proxy < best_proxy:
                        best_clean = current_best
                        best_proxy = current_proxy
                        best_label += "+diffopt"
                        print(f"  [{mode}] diff_opt improved: {best_proxy:.4f}",
                              flush=True)
                except Exception as e:
                    print(f"  [{mode}] diff_opt failed: {e}", flush=True)

            # Step 7: ABU5-targeted congestion shifting
            if self.USE_ABU5_SHIFT:
                try:
                    from dreamplace_integration.abu5_shifter import abu5_shift
                    shifted_pos, shift_info = abu5_shift(
                        best_clean, benchmark, self._get_plc(bench_dir),
                        max_iters=self.ABU5_MAX_ITERS,
                        time_budget_s=self.ABU5_TIME_BUDGET_S,
                        verbose=False,
                    )
                    if shift_info["improvement"] > 0:
                        from dreamplace_integration.check_overlaps import check_overlaps_torch
                        if check_overlaps_torch(shifted_pos, benchmark, margin=0.0) == 0:
                            best_clean = shifted_pos
                            best_proxy = shift_info["final_proxy"]
                            best_label += f"+abu5({shift_info['accepted_moves']}mv)"
                except Exception as e:
                    print(f"  [{mode}] abu5_shift failed: {e}", flush=True)

            return best_clean, best_proxy, best_label

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_plc(self, bench_dir: str):
        """Get a PlacementCost object for the given benchmark dir."""
        from macro_place.loader import load_benchmark_from_dir
        _, plc = load_benchmark_from_dir(bench_dir)
        return plc

    def _orientation_search(self, placement, benchmark, bench_dir, current_proxy):
        """
        Try flipping macro orientations (N→FN, N→FS, N→S) for top-K macros.
        Keep any flip that reduces proxy. Greedy per-macro.
        
        Klein-4 orientations: N (normal), FN (flip-north = mirror Y),
        FS (flip-south = mirror X+Y), S (south = mirror X)
        For placement, flipping orientation means reflecting pin offsets.
        Since we use macro centers (no pin offsets in diff_opt), orientation
        flips only matter if the evaluator uses pin positions.
        
        Simplified: try mirroring macro position around its center
        (swap x-offset of pins). This is equivalent to orientation flip
        for the TILOS evaluator.
        """
        from macro_place.objective import compute_proxy_cost as _cpc
        
        # For now, orientation flips don't change macro center positions
        # They only affect pin offsets. Since our optimizer doesn't use pin
        # offsets, orientation search won't help until we add pin offset support.
        # Return unchanged.
        return placement, current_proxy

    @staticmethod
    def _positions_dict_to_tensor(positions: dict, benchmark: Benchmark) -> torch.Tensor:
        """Map DREAMPlace name->(cx,cy) dict to TILOS placement tensor."""
        placement = benchmark.macro_positions.clone().float()
        n_found = 0
        for i in range(benchmark.num_macros):
            nm = benchmark.macro_names[i].replace(" ", "_").replace("/", "_")
            if nm in positions:
                cx, cy = positions[nm]
                placement[i, 0] = cx
                placement[i, 1] = cy
                n_found += 1
        if n_found < benchmark.num_macros // 2:
            raise RuntimeError(
                f"only mapped {n_found}/{benchmark.num_macros} macros from output"
            )
        return placement

    @staticmethod
    def _proxy(placement: torch.Tensor, bench_dir: str) -> float:
        """Compute proxy cost for a (validated) placement on the given benchmark."""
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_proxy_cost
        bm, plc = load_benchmark_from_dir(bench_dir)
        return float(compute_proxy_cost(placement, bm, plc)["proxy_cost"])

    def _scan_checkpoints(self, ckpt_dir: str, benchmark: Benchmark,
                          bench_dir: str, clean_overlaps_fn):
        """
        Walk all checkpoint .npz files, run cleanup on each, return the
        (placement, proxy, iter) with best proxy. Returns (None, None, -1)
        if no usable checkpoint is found.
        """
        import glob
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_proxy_cost

        meta_path = os.path.join(ckpt_dir, "meta.npz")
        if not os.path.isfile(meta_path):
            return None, None, -1
        meta = np.load(meta_path, allow_pickle=False)
        if "scale_factor" not in meta.files:
            return None, None, -1
        node_names = meta["node_names"]
        nmov = int(meta["num_movable_nodes"])
        ntot = int(meta["num_nodes"])
        scale_factor = float(meta["scale_factor"])
        shift = meta["shift_factor"] if "shift_factor" in meta.files else np.array([0.0, 0.0])
        inv = 1.0 / scale_factor if scale_factor != 0 else 1.0

        # Pre-build name → benchmark index lookup
        name_to_idx = {}
        for i in range(benchmark.num_macros):
            nm = benchmark.macro_names[i].replace(" ", "_").replace("/", "_")
            name_to_idx[nm] = i
        sizes = benchmark.macro_sizes.float()

        # Reuse a single benchmark/plc for proxy calls
        bm, plc = load_benchmark_from_dir(bench_dir)

        # Skip the first ~50% of iterations (early ones haven't converged enough
        # for cleanup to give a meaningful proxy). Pick from iter 300 onward to
        # keep cleanup work bounded.
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.npz")))
        # Heuristic: only scan checkpoints with iteration >= MIN_CHECKPOINT_ITER
        # to skip the divergent early ones.
        ckpt_paths = [
            p for p in ckpt_paths
            if int(os.path.basename(p).replace("ckpt_", "").replace(".npz", ""))
                >= self.MIN_CHECKPOINT_ITER
        ]
        # Adaptive stride based on benchmark size — keep cleanup cost bounded
        n_hard = benchmark.num_hard_macros
        stride = self.CHECKPOINT_SCAN_STRIDE
        if n_hard > 250:
            stride = max(stride, 4)
        if n_hard > 280:
            stride = max(stride, 6)
        if stride > 1:
            ckpt_paths = ckpt_paths[::stride]

        best_pos = None
        best_proxy = None
        best_iter = -1
        scan_start = time.time()

        for cp in ckpt_paths:
            if time.time() - scan_start > self.CHECKPOINT_SCAN_TIME_S:
                break
            d = np.load(cp, allow_pickle=False)
            pos_flat = d["pos"]
            it = int(d["iteration"])

            x_int = pos_flat[:ntot]
            y_int = pos_flat[ntot:2 * ntot]
            x_um = (x_int * inv + float(shift[0])) / 1000.0
            y_um = (y_int * inv + float(shift[1])) / 1000.0

            placement = benchmark.macro_positions.clone().float()
            for i in range(nmov):
                nm = str(node_names[i])
                if nm not in name_to_idx:
                    continue
                j = name_to_idx[nm]
                placement[j, 0] = float(x_um[i]) + float(sizes[j, 0]) / 2
                placement[j, 1] = float(y_um[i]) + float(sizes[j, 1]) / 2

            # Cleanup; skip if it can't produce a clean placement
            clean_pos, info = clean_overlaps_fn(
                placement, benchmark,
                legal_max_sweeps=300, jiggle_attempts=5,
                force_max_iterations=500, margin=0.0, verbose=False,
            )
            if info.n_overlaps_out > 0:
                continue

            p = float(compute_proxy_cost(clean_pos, bm, plc)["proxy_cost"])
            if best_proxy is None or p < best_proxy:
                best_pos = clean_pos
                best_proxy = p
                best_iter = it

        return best_pos, best_proxy, best_iter

    def _fallback_place(self, benchmark: Benchmark) -> torch.Tensor:
        if self._fallback is None:
            # Last resort: return the CT initial.plc (zero-overlap by construction)
            print(f"[placer_dreamplace] no fallback placer; returning CT initial.plc",
                  flush=True)
            return benchmark.macro_positions.clone().float()
        return self._fallback.place(benchmark)

    def _infer_benchmark_dir(self, benchmark: Benchmark) -> Optional[str]:
        """
        Locate the directory containing netlist.pb.txt / initial.plc for
        this benchmark. Tries common layouts on both local and eval
        environments.
        """
        candidates = [
            # Eval Docker layout (Dockerfile copies into /challenge/)
            f"/challenge/external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}",
            # Local repo layout
            os.path.join(
                _HERE, "..", "..", "external", "MacroPlacement",
                "Testcases", "ICCAD04", benchmark.name,
            ),
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "netlist.pb.txt")):
                return c
        return None
