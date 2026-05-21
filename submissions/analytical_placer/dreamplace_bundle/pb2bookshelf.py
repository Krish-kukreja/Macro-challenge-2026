"""
Convert TILOS MacroPlacement .pb.txt + .plc → DREAMPlace Bookshelf format.

Usage:
    python3 pb2bookshelf.py <benchmark_dir> <output_dir>

Example:
    python3 pb2bookshelf.py \
        external/MacroPlacement/Testcases/ICCAD04/ibm01 \
        /tmp/dp_ibm01

Output files:
    <output_dir>/<name>.nodes   - node sizes
    <output_dir>/<name>.nets    - net connectivity
    <output_dir>/<name>.pl      - initial placement
    <output_dir>/<name>.scl     - row structure
    <output_dir>/<name>.wts     - net weights
    <output_dir>/<name>.aux     - index file
    <output_dir>/<name>.json    - DREAMPlace config
"""

import sys
import os
import math

# Add project root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from macro_place.loader import load_benchmark_from_dir


SCALE = 1000  # μm → integer DBU multiplier for Bookshelf format


def _i(x):
    """Round to integer for Bookshelf output."""
    return int(round(float(x) * SCALE))


def convert(benchmark_dir: str, output_dir: str, gpu: int = 1,
            iterations: int = 2000, target_density: float = None,
            mode: str = "default", override_params: dict = None) -> str:
    """
    Convert a TILOS benchmark to DREAMPlace Bookshelf format.

    All geometry is scaled by SCALE (default 1000) to satisfy DREAMPlace's
    requirement of integer coordinates in the .scl row file. Read back
    output via read_dreamplace_output() which divides by SCALE.

    mode:
        "default"          — vanilla global placement (v3/v4 baseline)
        "congestion_aware" — routability_opt_flag=1, RUDY area adjustment,
                             pin/route capacities from benchmark routing info.
                             Expected to lower congestion at small density cost.
        "fine_grid"        — 1024×1024 bins (vs 512), tighter overflow stop,
                             more iterations. Slower but better convergence
                             on dense placements.
        "two_pass"         — same as default for stage 1; the caller is
                             expected to follow up with a congestion_aware
                             second pass using --bookshelf_pl_input.

    Returns the path to the generated .json config file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load benchmark via TILOS loader
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)
    name = benchmark.name

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]

    # DREAMPlace treats everything as "cells" — macros are fixed-size cells,
    # ports are fixed terminals. We place all hard macros as movable cells
    # and ports as fixed terminals.

    # ---- .nodes file ----
    nodes_path = os.path.join(output_dir, f"{name}.nodes")
    with open(nodes_path, 'w') as f:
        f.write("UCLA nodes 1.0\n\n")
        total_nodes = n_macros + n_ports
        f.write(f"NumNodes : {total_nodes}\n")
        f.write(f"NumTerminals : {n_ports}\n\n")

        # Hard macros (movable)
        for i in range(n_hard):
            w = _i(benchmark.macro_sizes[i, 0])
            h = _i(benchmark.macro_sizes[i, 1])
            name_i = benchmark.macro_names[i].replace(' ', '_').replace('/', '_')
            f.write(f"\t{name_i}\t{w}\t{h}\n")

        # Soft macros (movable, treated as standard cells)
        for i in range(n_hard, n_macros):
            w = _i(benchmark.macro_sizes[i, 0])
            h = _i(benchmark.macro_sizes[i, 1])
            name_i = benchmark.macro_names[i].replace(' ', '_').replace('/', '_')
            f.write(f"\t{name_i}\t{w}\t{h}\n")

        # Ports (fixed terminals)
        for i in range(n_ports):
            f.write(f"\tport_{i}\t1\t1\tterminal\n")

    # ---- .pl file (initial placement) ----
    pl_path = os.path.join(output_dir, f"{name}.pl")
    with open(pl_path, 'w') as f:
        f.write("UCLA pl 1.0\n\n")

        # Hard macros — use initial.plc positions (center → bottom-left)
        for i in range(n_hard):
            cx = float(benchmark.macro_positions[i, 0])
            cy = float(benchmark.macro_positions[i, 1])
            w = float(benchmark.macro_sizes[i, 0])
            h = float(benchmark.macro_sizes[i, 1])
            # Bookshelf uses bottom-left corner
            bx = _i(cx - w / 2)
            by = _i(cy - h / 2)
            name_i = benchmark.macro_names[i].replace(' ', '_').replace('/', '_')
            fixed = benchmark.macro_fixed[i].item()
            orient = "N"
            if fixed:
                f.write(f"\t{name_i}\t{bx}\t{by}\t:\t{orient}\t/FIXED\n")
            else:
                f.write(f"\t{name_i}\t{bx}\t{by}\t:\t{orient}\n")

        # Soft macros
        for i in range(n_hard, n_macros):
            cx = float(benchmark.macro_positions[i, 0])
            cy = float(benchmark.macro_positions[i, 1])
            w = float(benchmark.macro_sizes[i, 0])
            h = float(benchmark.macro_sizes[i, 1])
            bx = _i(cx - w / 2)
            by = _i(cy - h / 2)
            name_i = benchmark.macro_names[i].replace(' ', '_').replace('/', '_')
            f.write(f"\t{name_i}\t{bx}\t{by}\t:\tN\n")

        # Ports (fixed at their positions)
        for i in range(n_ports):
            px = _i(benchmark.port_positions[i, 0])
            py = _i(benchmark.port_positions[i, 1])
            f.write(f"\tport_{i}\t{px}\t{py}\t:\tN\t/FIXED\n")

    # ---- .nets file ----
    nets_path = os.path.join(output_dir, f"{name}.nets")
    # Build name lookup
    node_names = []
    for i in range(n_macros):
        node_names.append(benchmark.macro_names[i].replace(' ', '_').replace('/', '_'))
    for i in range(n_ports):
        node_names.append(f"port_{i}")

    with open(nets_path, 'w') as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {benchmark.num_nets}\n")
        total_pins = sum(len(benchmark.net_nodes[i]) for i in range(benchmark.num_nets))
        f.write(f"NumPins : {total_pins}\n\n")

        for net_id in range(benchmark.num_nets):
            nodes = benchmark.net_nodes[net_id].tolist()
            w = float(benchmark.net_weights[net_id])
            f.write(f"NetDegree : {len(nodes)} net{net_id}\n")
            for j, node_idx in enumerate(nodes):
                nname = node_names[node_idx]
                # First node is driver (O), rest are sinks (I)
                direction = "O" if j == 0 else "I"
                f.write(f"\t{nname}\t{direction} : 0.0 0.0\n")

    # ---- .wts file ----
    wts_path = os.path.join(output_dir, f"{name}.wts")
    with open(wts_path, 'w') as f:
        f.write("UCLA wts 1.0\n\n")
        for net_id in range(benchmark.num_nets):
            w = float(benchmark.net_weights[net_id])
            if abs(w - 1.0) > 1e-6:
                f.write(f"\tnet{net_id}\t{w:.4f}\n")

    # ---- .scl file (row structure) ----
    # DREAMPlace requires INTEGER coordinates in .scl. We've scaled all geometry
    # by SCALE so canvas is now SCALE*cw x SCALE*ch in DBU.
    scl_path = os.path.join(output_dir, f"{name}.scl")

    grid_rows = benchmark.grid_rows
    # row_height is canvas_height / grid_rows in microns; in DBU = SCALE*ch / grid_rows
    # Pick an integer row height so rows tile the canvas exactly.
    canvas_h_dbu = _i(ch)
    canvas_w_dbu = _i(cw)
    # Use largest integer row_height that fits grid_rows times into canvas_h_dbu
    row_height_dbu = max(1, canvas_h_dbu // grid_rows)
    site_width_dbu = max(1, canvas_w_dbu // benchmark.grid_cols)
    num_sites = canvas_w_dbu // site_width_dbu

    with open(scl_path, 'w') as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {grid_rows}\n\n")
        for r in range(grid_rows):
            y = r * row_height_dbu
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {y}\n")
            f.write(f"  Height        : {row_height_dbu}\n")
            f.write(f"  Sitewidth     : {site_width_dbu}\n")
            f.write(f"  Sitespacing   : {site_width_dbu}\n")
            f.write(f"  Siteorient    : N\n")
            f.write(f"  Sitesymmetry  : Y\n")
            f.write(f"  SubrowOrigin  : 0  NumSites : {num_sites}\n")
            f.write("End\n")

    # ---- .aux file ----
    aux_path = os.path.join(output_dir, f"{name}.aux")
    with open(aux_path, 'w') as f:
        f.write(f"RowBasedPlacement : {name}.nodes {name}.nets {name}.wts "
                f"{name}.pl {name}.scl\n")

    # ---- DREAMPlace JSON config (mode-dependent) ----
    if target_density is None:
        from macro_place.objective import compute_proxy_cost as _cpc
        ct_costs = _cpc(benchmark.macro_positions.float(), benchmark, plc)
        ct_density = float(ct_costs["density_cost"])

        if ct_density < 0.80:
            target_density = max(0.50, ct_density * 0.95)
        else:
            total_movable_area = sum(
                float(benchmark.macro_sizes[i, 0]) * float(benchmark.macro_sizes[i, 1])
                for i in range(n_macros)
            )
            canvas_area = cw * ch
            util = total_movable_area / canvas_area
            target_density = min(0.95, max(util * 1.05, 0.85))

    # Mode-specific knobs
    num_bins = 512
    iters = iterations
    stop_overflow = 0.10
    routability_opt = 0
    enable_fillers = 1
    use_capacity_from_benchmark = False
    random_seed = 1000

    if mode == "fine_grid":
        num_bins = 1024
        iters = max(iters, 3000)
        stop_overflow = 0.05
    elif mode == "congestion_aware":
        # Soft congestion mode (v5 proven): tighter density convergence
        # + 2x density_weight. No routability_opt (NCTUgr hurts more than helps).
        stop_overflow = 0.05
        use_capacity_from_benchmark = False
    elif mode == "default_seed2":
        # Same as default but different seed — gives DREAMPlace a different
        # initial-noise trajectory; useful when the default seed lands in a
        # bad local optimum.
        random_seed = 2000
    elif mode == "default_seed3":
        random_seed = 3000
    elif mode == "gift_init":
        # GiFt initialization — graph-filter-based placement init
        # that can improve convergence quality.
        pass  # gift_init_flag handled below
    elif mode == "override":
        # Per-benchmark override from focused sweep results.
        # override_params dict has: congestion_weight, density_weight, seed
        if override_params:
            random_seed = override_params.get("seed", 1000)
            stop_overflow = 0.07  # tighter for focused configs
    elif mode == "two_pass":
        # Same as default; caller will set bookshelf_pl_input on the second pass
        pass

    mode_density_weight = 8e-5
    if mode == "congestion_aware":
        mode_density_weight = 1.6e-4  # 2x default — proven in v5
    elif mode == "fine_grid":
        mode_density_weight = 8e-5
    elif mode == "override" and override_params:
        mode_density_weight = override_params.get("density_weight", 8e-5)

    # Optional capacity overrides for congestion-aware mode
    h_cap = float(getattr(benchmark, "hroutes_per_micron", 65.96))
    v_cap = float(getattr(benchmark, "vroutes_per_micron", 106.96))
    # DREAMPlace expects unit_horizontal_capacity per "unit distance"
    # (typically row-height units). We pass the TILOS values normalized.
    # Keep defaults if unsure (DREAMPlace's defaults are 1.5625 / 1.45).
    if not use_capacity_from_benchmark:
        h_cap = 1.5625
        v_cap = 1.45

    json_path = os.path.join(output_dir, f"{name}.json")
    cfg_lines = [
        '{',
        f'"aux_input" : "{os.path.abspath(aux_path)}",',
        f'"gpu" : {gpu},',
        f'"num_bins_x" : {num_bins},',
        f'"num_bins_y" : {num_bins},',
        '"global_place_stages" : [{',
        f'  "num_bins_x" : {num_bins},',
        f'  "num_bins_y" : {num_bins},',
        f'  "iteration" : {iters},',
        '  "learning_rate" : 0.01,',
        '  "wirelength" : "weighted_average",',
        '  "optimizer" : "nesterov",',
        '  "Llambda_density_weight_iteration" : 1,',
        '  "Lsub_iteration" : 1',
        '}],',
        f'"target_density" : {target_density:.4f},',
        f'"density_weight" : {mode_density_weight:.2e},',
        f'"random_seed" : {random_seed},',
        '"scale_factor" : 1.0,',
        '"ignore_net_degree" : 100,',
        f'"enable_fillers" : {enable_fillers},',
        '"gp_noise_ratio" : 0.025,',
        '"global_place_flag" : 1,',
        '"legalize_flag" : 0,',
        '"detailed_place_flag" : 0,',
        f'"stop_overflow" : {stop_overflow:.3f},',
        '"dtype" : "float32",',
        '"random_center_init_flag" : 0,',
        '"deterministic_flag" : 1,',
        '"num_threads" : 8,',
        '"macro_place_flag" : 1,',
        '"use_bb" : 1,',
        f'"routability_opt_flag" : {routability_opt},',
        f'"node_area_adjust_overflow" : {0.4 if routability_opt else 0.15},',
        '"adjust_rudy_area_flag" : 1,',
        '"adjust_pin_area_flag" : 1,',
        f'"unit_horizontal_capacity" : {h_cap},',
        f'"unit_vertical_capacity" : {v_cap},',
        f'"gift_init_flag" : {1 if mode == "gift_init" else 0},',
        '"gift_init_scale" : 1.0,',
        f'"result_dir" : "{os.path.join(output_dir, "results")}"',
        '}',
    ]
    with open(json_path, 'w') as f:
        f.write('\n'.join(cfg_lines) + '\n')

    print(f"[pb2bookshelf] {name}: {n_hard} hard macros, {n_soft} soft macros, "
          f"{n_ports} ports, {benchmark.num_nets} nets")
    print(f"  Canvas: {cw:.2f} x {ch:.2f}  target_density={target_density:.3f}")
    print(f"  Output: {json_path}")
    return json_path


def read_dreamplace_output(pl_path: str, benchmark_dir: str):
    """
    Read DREAMPlace output .pl file and return macro positions as a dict.

    Returns: dict mapping macro_name -> (center_x, center_y)
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from macro_place.loader import load_benchmark_from_dir

    benchmark, _ = load_benchmark_from_dir(benchmark_dir)

    # Build name → size lookup
    name_to_size = {}
    for i in range(benchmark.num_macros):
        n = benchmark.macro_names[i].replace(' ', '_').replace('/', '_')
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        name_to_size[n] = (w, h)

    positions = {}
    with open(pl_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('UCLA') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            node_name = parts[0]
            if node_name.startswith('port_'):
                continue
            try:
                bx = float(parts[1])
                by = float(parts[2])
            except ValueError:
                continue
            if node_name in name_to_size:
                w, h = name_to_size[node_name]
                # Convert from scaled DBU back to microns, bottom-left → center
                positions[node_name] = (bx / SCALE + w / 2, by / SCALE + h / 2)

    return positions


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 pb2bookshelf.py <benchmark_dir> <output_dir> [gpu=1]")
        sys.exit(1)

    benchmark_dir = sys.argv[1]
    output_dir = sys.argv[2]
    gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    json_path = convert(benchmark_dir, output_dir, gpu=gpu)
    print(f"\nRun DREAMPlace with:")
    print(f"  cd ~/DREAMPlace/install && python3 dreamplace/Placer.py {json_path}")
