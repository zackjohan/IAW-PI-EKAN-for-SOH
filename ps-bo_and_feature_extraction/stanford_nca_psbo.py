"""stanford_nca_psbo: native dataset processing and retained experiments."""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

if __package__:
    from .psbo_core import (
        safe_mkdir,
        longest_contiguous_segment,
        downsample_to_hz,
        evaluate_window,
        psbo_optimize,
        plot_2x2,
        save_figure_3formats,
        extract_features_csv,
        write_window_report,
        assert_reference_held_out,
        apply_bias_drift_to_current,
        apply_init_error_to_soc,
        apply_random_walk_to_soc,
        apply_spike_noise_to_soc,
    )
    from .psbo_core import run_dataset_cli
else:
    from psbo_core import (
        safe_mkdir,
        longest_contiguous_segment,
        downsample_to_hz,
        evaluate_window,
        psbo_optimize,
        plot_2x2,
        save_figure_3formats,
        extract_features_csv,
        write_window_report,
        assert_reference_held_out,
        apply_bias_drift_to_current,
        apply_init_error_to_soc,
        apply_random_walk_to_soc,
        apply_spike_noise_to_soc,
    )
    from psbo_core import run_dataset_cli
STANFORD_DIR = "D:\\Feature Extarction\\Stanford\\Data"
DIAG_CAP_DIR = "D:\\Feature Extarction\\Stanford\\Data\\Diagnostic_Capacities"
OUT_DIR = "D:\\Feature Extarction\\Stanford\\PS-BO Window Selected"
MODULES_ROOT = "D:\\Feature Extarction\\Stanford\\PSBO_Modules"
MODULE_DIRS = {
    "M6_sampling_frequency": os.path.join(MODULES_ROOT, "06_Sampling_Frequency"),
    "M8_robustness": os.path.join(MODULES_ROOT, "08_Robustness"),
}
DATASET_NAME = "Stanford"
REFERENCE_CELL = "Cell_059.csv"
MODEL_SPLITS = {"test": ["Cell_064", "Cell_074", "Cell_096"], "val": [], "train": []}
ENFORCE_LEAKAGE_GUARD = True
FIXED_WINDOWS = [
    (0, 100, "D:\\Feature Extarction\\Stanford\\0 %- 100% Window Selected"),
    (20, 80, "D:\\Feature Extarction\\Stanford\\0 %- 80% Window Selected"),
    (25, 75, "D:\\Feature Extarction\\Stanford\\25 %- 75% Window Selected"),
    (30, 70, "D:\\Feature Extarction\\Stanford\\30 %- 70% Window Selected"),
    (40, 60, "D:\\Feature Extarction\\Stanford\\40 %- 60% Window Selected"),
]
STANFORD_CELLS = [
    "Cell_059.csv",
    "Cell_060.csv",
    "Cell_063.csv",
    "Cell_064.csv",
    "Cell_069.csv",
    "Cell_070.csv",
    "Cell_074.csv",
    "Cell_078.csv",
    "Cell_079.csv",
    "Cell_081.csv",
    "Cell_082.csv",
    "Cell_088.csv",
    "Cell_091.csv",
    "Cell_092.csv",
    "Cell_095.csv",
    "Cell_096.csv",
]
SUMMARY_FILES = {
    f"Cell_{n}": f"Publishing_data_aging_summary_cell_{n}.csv"
    for n in [
        "059",
        "060",
        "063",
        "064",
        "069",
        "070",
        "074",
        "078",
        "079",
        "081",
        "082",
        "088",
        "091",
        "092",
        "095",
        "096",
    ]
}
DIAG_FILES = {
    f"Cell_{n}": f"cell_{n}_diagnostic_capacities.csv"
    for n in [
        "059",
        "060",
        "063",
        "064",
        "069",
        "070",
        "074",
        "078",
        "079",
        "081",
        "082",
        "088",
        "091",
        "092",
        "095",
        "096",
    ]
}
DIAG_CAP_COL = "C/2 discharge CCCV Capacity (normalized)"
CAP_COL_SUMMARY = "Normalized Discharge Capacity [-]"
CAP_THRESHOLD = 1.5
COL_TIME = "Test (Sec)"
COL_CURR = "Normalized Current (C-rate)"
COL_VOLT = "Volts"
COL_CYC = "Cyc#"
COL_STEP = "Step"
COL_STATE = "State"
CHARGE_CURRENT_THRESHOLD = 0.0001
LAST_COL_NAME = "Normalized Capacity"
SCENARIO_CONFIGS = [
    {
        "name": "Scenario A",
        "label": "Bias Drift",
        "type": "bias_drift",
        "params": {"bias_crate": 0.02},
    },
    {
        "name": "Scenario B",
        "label": "Init Error",
        "type": "init_error",
        "params": {"init_error_percent": 5.0},
    },
    {
        "name": "Scenario C",
        "label": "Random Walk",
        "type": "random_walk",
        "params": {"sigma_step": 0.05},
    },
    {
        "name": "Scenario D",
        "label": "Spike Noise",
        "type": "spike_noise",
        "params": {"spike_prob": 0.02, "spike_mag": 15.0},
    },
]
SCENARIO_RANDOM_SEED = 42
SAMPLING_SWEEP_HZ = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
NATIVE_SOC_MODE = "absolute"


def _detect_cycle_column(df: pd.DataFrame) -> str:
    candidates = [
        "Cycle",
        "Cyc#",
        "cycle",
        "CYC",
        "cyc",
        "Cycle #",
        "Cycle_Number",
        "Cycle number",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            if df[c].nunique(dropna=True) > 10:
                return c
    raise KeyError("Could not detect cycle column in aging summary file.")


def load_capacity_map(cell_name_no_ext: str) -> dict:
    if cell_name_no_ext not in SUMMARY_FILES:
        raise KeyError(f"No summary mapping provided for {cell_name_no_ext}")
    summary_path = os.path.join(STANFORD_DIR, SUMMARY_FILES[cell_name_no_ext])
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Aging summary file not found: {summary_path}")
    df = pd.read_csv(summary_path, low_memory=False)
    if CAP_COL_SUMMARY not in df.columns:
        raise KeyError(f"Missing '{CAP_COL_SUMMARY}' in {summary_path}")
    cyc_col = _detect_cycle_column(df)
    cyc = pd.to_numeric(df[cyc_col], errors="coerce").astype("Int64")
    cap = pd.to_numeric(df[CAP_COL_SUMMARY], errors="coerce")
    valid = cyc.notna() & cap.notna() & (cap <= CAP_THRESHOLD)
    df2 = pd.DataFrame(
        {"cycle": cyc[valid].astype(int), "cap": cap[valid].astype(float)}
    )
    df2 = df2.drop_duplicates(subset=["cycle"], keep="last")
    return dict(zip(df2["cycle"].to_numpy(), df2["cap"].to_numpy()))


def load_diagnostic_interpolator(cell_name_no_ext: str):
    if cell_name_no_ext not in DIAG_FILES:
        return None
    diag_path = os.path.join(DIAG_CAP_DIR, DIAG_FILES[cell_name_no_ext])
    if not os.path.exists(diag_path):
        tqdm.write(
            f"  WARNING: Diagnostic file not found for {cell_name_no_ext}: {diag_path}"
        )
        return None
    try:
        df = pd.read_csv(diag_path, low_memory=False)
    except Exception as e:
        tqdm.write(
            f"  WARNING: Could not read diagnostic file for {cell_name_no_ext}: {e}"
        )
        return None
    cycle_candidates = [
        "Cycle",
        "Cyc#",
        "cycle",
        "CYC",
        "cyc",
        "Cycle #",
        "Cycle_Number",
        "Cycle number",
    ]
    cyc_col = None
    for c in cycle_candidates:
        if c in df.columns:
            cyc_col = c
            break
    if cyc_col is None:
        if df.shape[1] >= 2:
            cyc_col = df.columns[1]
        else:
            tqdm.write(f"  WARNING: Cannot detect cycle column in {diag_path}")
            return None
    if DIAG_CAP_COL not in df.columns:
        col_lower = {c.lower().strip(): c for c in df.columns}
        key = DIAG_CAP_COL.lower().strip()
        if key in col_lower:
            cap_col = col_lower[key]
        else:
            tqdm.write(
                f"  WARNING: '{DIAG_CAP_COL}' not found in {diag_path}.\n           Available columns: {list(df.columns)}"
            )
            return None
    else:
        cap_col = DIAG_CAP_COL
    cyc = pd.to_numeric(df[cyc_col], errors="coerce")
    cap = pd.to_numeric(df[cap_col], errors="coerce")
    valid = cyc.notna() & cap.notna() & np.isfinite(cyc) & np.isfinite(cap)
    cyc_v = cyc[valid].to_numpy(dtype=float)
    cap_v = cap[valid].to_numpy(dtype=float)
    if len(cyc_v) < 2:
        tqdm.write(
            f"  WARNING: Fewer than 2 valid diagnostic points for {cell_name_no_ext} — skipping smoothing."
        )
        return None
    order = np.argsort(cyc_v)
    cyc_v, cap_v = (cyc_v[order], cap_v[order])
    _, unique_idx = np.unique(cyc_v, return_index=True)
    cyc_v, cap_v = (cyc_v[unique_idx], cap_v[unique_idx])
    pchip = PchipInterpolator(cyc_v, cap_v, extrapolate=False)
    cyc_min = float(cyc_v[0])
    cyc_max = float(cyc_v[-1])
    cap_lo = float(cap_v[-1])
    cap_hi = float(cap_v[0])

    def smooth_interp(cycle_number: float) -> float:
        c = float(cycle_number)
        if c <= cyc_min:
            return cap_hi
        if c >= cyc_max:
            return cap_lo
        val = float(pchip(c))
        return float(np.clip(val, 0.0, CAP_THRESHOLD))

    return smooth_interp


class StanfordCellAging:
    """
    Charge-segment extraction. Identical to the original when called with
    defaults (target_hz=1.0, soc_mode='absolute', noise_config=None).
    """

    def __init__(
        self,
        csv_path: str,
        noise_config: dict = None,
        noise_seed: int = 42,
        target_hz: float = 1.0,
    ):
        self.path = csv_path
        self.name = os.path.basename(csv_path).replace(".csv", "")
        self.soc_mode = NATIVE_SOC_MODE
        self.target_hz = target_hz
        df = pd.read_csv(csv_path, low_memory=False)
        required = [COL_TIME, COL_CURR, COL_VOLT, COL_CYC, COL_STEP, COL_STATE]
        for c in required:
            if c not in df.columns:
                raise KeyError(f"Missing required column '{c}' in {csv_path}")
        grouped = df.groupby(COL_CYC, sort=True)
        self.charge_segments = []
        self.cycle_ids = []
        for cyc_id, g in grouped:
            if int(cyc_id) <= 2:
                continue
            t_all = g[COL_TIME].to_numpy(dtype=float)
            curr_all = g[COL_CURR].to_numpy(dtype=float)
            V_all = g[COL_VOLT].to_numpy(dtype=float)
            charge_mask = curr_all > CHARGE_CURRENT_THRESHOLD
            charge_idx = np.where(charge_mask)[0]
            if len(charge_idx) == 0:
                continue
            longest_idx = longest_contiguous_segment(charge_idx)
            if len(longest_idx) < 10:
                continue
            t_raw = t_all[longest_idx]
            V_raw = V_all[longest_idx]
            curr_raw = curr_all[longest_idx]
            t_raw = t_raw - t_raw[0]
            keep = downsample_to_hz(t_raw, target_hz=target_hz)
            if len(keep) < 30:
                continue
            t_c = t_raw[keep]
            V_c = V_raw[keep]
            curr_c = curr_raw[keep]
            if noise_config is not None and noise_config["type"] == "bias_drift":
                curr_c = apply_bias_drift_to_current(
                    curr_c, bias_crate=noise_config["params"]["bias_crate"]
                )
            dt = np.empty_like(t_c)
            dt[0] = 0.0
            dt[1:] = np.diff(t_c)
            dQ = curr_c * dt / 3600.0
            cum = np.cumsum(dQ)
            soc = cum * 100.0
            if soc[-1] < 5.0:
                continue
            if noise_config is not None and noise_config["type"] != "bias_drift":
                local_seed = int(
                    noise_seed + int(cyc_id) * 1000 + sum((ord(ch) for ch in self.name))
                )
                rng = np.random.default_rng(local_seed)
                ntype = noise_config["type"]
                params = noise_config["params"]
                if ntype == "init_error":
                    soc = apply_init_error_to_soc(
                        soc, init_error_percent=params["init_error_percent"]
                    )
                elif ntype == "random_walk":
                    soc = apply_random_walk_to_soc(
                        soc, sigma_step=params["sigma_step"], rng=rng
                    )
                elif ntype == "spike_noise":
                    soc = apply_spike_noise_to_soc(
                        soc,
                        spike_prob=params["spike_prob"],
                        spike_mag=params["spike_mag"],
                        rng=rng,
                    )
            self.charge_segments.append(
                {
                    "soc": soc.astype(float),
                    "V": V_c.astype(float),
                    "t": t_c.astype(float),
                }
            )
            self.cycle_ids.append(int(cyc_id))
        self.n_cycles = len(self.charge_segments)


def cell_path(fn):
    return os.path.join(STANFORD_DIR, fn if fn.endswith(".csv") else fn + ".csv")


def build_cell(fn, **kw):
    return StanfordCellAging(cell_path(fn), **kw)


def cap_map_of(fn):
    return load_capacity_map(os.path.basename(str(fn)).replace(".csv", ""))


def extract_all_cells(lower, upper, out_dir, noise_config=None, target_hz=1.0, desc=""):
    """Extract features for every cell into out_dir. CSV naming unchanged."""
    safe_mkdir(out_dir)
    n_csv = n_rows = 0
    for fn in tqdm(STANFORD_CELLS, desc=desc or f"[{lower}%-{upper}%]"):
        p = cell_path(fn)
        if not os.path.exists(p):
            tqdm.write(f"WARNING: Missing cell file: {p}")
            continue
        cell_name = os.path.basename(p).replace(".csv", "")
        try:
            cap_map = load_capacity_map(cell_name)
            cell = StanfordCellAging(
                p,
                noise_config=noise_config,
                noise_seed=SCENARIO_RANDOM_SEED,
                target_hz=target_hz,
            )
            diag_interp = load_diagnostic_interpolator(cell_name)
            df_feat = extract_features_csv(
                cell,
                cap_map,
                lower,
                upper,
                diag_interp=diag_interp,
                cap_threshold=CAP_THRESHOLD,
                last_col_name=LAST_COL_NAME,
            )
            out_csv = os.path.join(out_dir, fn)
            df_feat.to_csv(out_csv, index=False)
            n_csv += 1
            n_rows += len(df_feat)
            tqdm.write(f"  {fn}: saved {len(df_feat)} rows -> {out_csv}")
        except Exception as e:
            tqdm.write(f"ERROR: {fn} -> {e}")
    return (n_csv, n_rows)


def _reference_setup():
    ref_path = cell_path(REFERENCE_CELL)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference cell not found: {ref_path}")
    ref_name = os.path.basename(ref_path).replace(".csv", "")
    if ENFORCE_LEAKAGE_GUARD:
        assert_reference_held_out(REFERENCE_CELL, MODEL_SPLITS, DATASET_NAME)
    cap_map_ref = load_capacity_map(ref_name)
    cell_ref = StanfordCellAging(ref_path)
    return (ref_name, cell_ref, cap_map_ref)


def option_1_psbo():
    print("\n" + "=" * 110)
    print("[1] PS-BO WINDOW SELECTION + FEATURE EXTRACTION")
    print("=" * 110)
    safe_mkdir(OUT_DIR)
    ref_name, cell_ref, cap_map_ref = _reference_setup()
    print(f"Reference: {cell_ref.name} | aging cycles extracted: {cell_ref.n_cycles}")
    print(
        f"Capacity points available (after filter <= {CAP_THRESHOLD}): {len(cap_map_ref)}\n"
    )
    best, score_map, axes, analysis = psbo_optimize(cell_ref, cap_map_ref)
    lo, hi = (int(best["lower_soc"]), int(best["upper_soc"]))
    write_window_report(
        os.path.join(OUT_DIR, f"{cell_ref.name}_PSBO_window_report.txt"),
        header_lines=[
            "DATASET: STANFORD (CSV)",
            "MODE: PER-AGING-CYCLE",
            "CAPACITY SOURCE: Publishing_data_aging_summary_cell_XXX.csv",
            f"CAPACITY COLUMN: {CAP_COL_SUMMARY}",
            f"CAP THRESHOLD: <= {CAP_THRESHOLD}",
            "SOC TYPE: ABSOLUTE (integrated from Normalized Current, start = 0 %)",
            f"CHARGE DETECTION: Normalized Current (C-rate) > {CHARGE_CURRENT_THRESHOLD}",
            f"DIAGNOSTIC EXCLUSION: cap_map join (cap > {CAP_THRESHOLD} excluded) + Cyc# <= 2 skipped",
            "CAPACITY SMOOTHING: PCHIP interpolation through diagnostic RPT anchors",
            f"  Diagnostic capacity column: {DIAG_CAP_COL}",
        ],
        best=best,
        analysis=analysis,
        cell=cell_ref,
    )
    print(f"\n[OK] Selected SOC window: {lo}% to {hi}%")
    fig = plot_2x2(
        best, score_map, axes, cell_ref, title_text=f"Stanford ({cell_ref.name})"
    )
    save_figure_3formats(
        fig, os.path.join(OUT_DIR, f"{cell_ref.name}_psbo_comprehensive")
    )
    print("[OK] Saved 2x2 figure (PNG/PDF/SVG @600 dpi)\n")
    extract_all_cells(lo, hi, OUT_DIR, desc=f"PS-BO window [{lo}%-{hi}%]")
    print(f"\n[OK] PS-BO extraction complete -> {OUT_DIR}")
    return {
        "best": best,
        "lower": lo,
        "upper": hi,
        "cell_ref": cell_ref,
        "cap_map_ref": cap_map_ref,
    }


def option_2_fixed_windows():
    print("\n" + "=" * 110)
    print("[2] FIXED SOC WINDOWS")
    print("=" * 110)
    for lo, hi, d in FIXED_WINDOWS:
        safe_mkdir(d)
        print(f"\n  --- Fixed Window: {lo}% - {hi}%  ->  {d} ---")
        extract_all_cells(lo, hi, d, desc=f"Fixed window [{lo}%-{hi}%]")
    print("\n[OK] All fixed windows complete.")


def option_3_sampling_frequency(psbo_result=None):
    print("\n" + "=" * 110)
    print("[3] SAMPLING-FREQUENCY SWEEP")
    print("=" * 110)
    out = MODULE_DIRS["M6_sampling_frequency"]
    safe_mkdir(out)
    if psbo_result is None:
        ref_name, cell_ref, cap_map_ref = _reference_setup()
        best, score_map, axes, analysis = psbo_optimize(cell_ref, cap_map_ref)
        lo, hi = (int(best["lower_soc"]), int(best["upper_soc"]))
    else:
        lo, hi = (psbo_result["lower"], psbo_result["upper"])
    print(f"Window held fixed at the PS-BO selection: {lo}%-{hi}%")
    print(f"Rates: {SAMPLING_SWEEP_HZ} Hz   (1.0 Hz reproduces the native run)\n")
    rows = []
    for hz in SAMPLING_SWEEP_HZ:
        d = os.path.join(out, f"{hz}Hz")
        print(f"\n  --- {hz} Hz  ->  {d} ---")
        n_csv, n_rows = extract_all_cells(lo, hi, d, target_hz=hz, desc=f"{hz} Hz")
        try:
            ref = build_cell(REFERENCE_CELL, target_hz=hz)
            res = evaluate_window(lo, hi, ref, cap_map_of(REFERENCE_CELL))
        except Exception:
            res = None
        rows.append(
            {
                "target_hz": hz,
                "lower_soc": lo,
                "upper_soc": hi,
                "csv_written": n_csv,
                "rows_written": n_rows,
                "window_score": res["score"] if res else np.nan,
                "coverage": res["coverage"] if res else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(out, "sampling_frequency_summary.csv"), index=False
    )
    print(f"\n[OK] -> {out}")
    print(
        "  Feed these folders to ablation_runner.py via WINDOW_FOLDERS (one entry per rate)."
    )


def option_4_robustness(psbo_result=None):
    print("\n" + "=" * 110)
    print("[4] ROBUSTNESS SCENARIOS A-D")
    print("=" * 110)
    out = MODULE_DIRS["M8_robustness"]
    safe_mkdir(out)
    if psbo_result is None:
        ref_name, cell_ref, cap_map_ref = _reference_setup()
        best, score_map, axes, analysis = psbo_optimize(cell_ref, cap_map_ref)
        clean_lo = int(best["lower_soc"])
        clean_hi = int(best["upper_soc"])
    else:
        cell_ref = psbo_result["cell_ref"]
        cap_map_ref = psbo_result["cap_map_ref"]
        clean_lo, clean_hi = (psbo_result["lower"], psbo_result["upper"])
    print(f"Clean-data PS-BO window: {clean_lo}%-{clean_hi}%")
    print("Two variants are produced per scenario:")
    print("  clean_window  — window fixed from clean data, features from noisy")
    print("                  data. Isolates FEATURE degradation. [headline]")
    print("  noisy_window  — window re-selected on the noisy reference cell.")
    print("                  Measures WINDOW-SELECTION stability.\n")
    rows = []
    for sc in SCENARIO_CONFIGS:
        sc_dir = os.path.join(
            out, f"{sc['name'].replace(' ', '_')}_{sc['label'].replace(' ', '_')}"
        )
        print("\n" + "-" * 100)
        print(f"  {sc['name']} — {sc['label']}  |  params: {sc['params']}")
        print("-" * 100)
        d_clean = os.path.join(sc_dir, "clean_window")
        print(f"\n  [a] clean_window {clean_lo}%-{clean_hi}%  ->  {d_clean}")
        n_csv_a, n_rows_a = extract_all_cells(
            clean_lo, clean_hi, d_clean, noise_config=sc, desc=f"{sc['name']} clean-win"
        )
        ref_noisy = build_cell(
            REFERENCE_CELL, noise_config=sc, noise_seed=SCENARIO_RANDOM_SEED
        )
        best_n, score_map_n, axes_n, an_n = psbo_optimize(
            ref_noisy, cap_map_ref, desc_suffix=f" [{sc['name']}]"
        )
        nlo, nhi = (int(best_n["lower_soc"]), int(best_n["upper_soc"]))
        d_noisy = os.path.join(sc_dir, "noisy_window")
        print(f"\n  [b] noisy_window {nlo}%-{nhi}%  ->  {d_noisy}")
        n_csv_b, n_rows_b = extract_all_cells(
            nlo, nhi, d_noisy, noise_config=sc, desc=f"{sc['name']} noisy-win"
        )
        fig = plot_2x2(
            best_n,
            score_map_n,
            axes_n,
            ref_noisy,
            title_text=f"Stanford ({ref_noisy.name}) — {sc['name']}: {sc['label']}",
        )
        save_figure_3formats(
            fig, os.path.join(sc_dir, f"{ref_noisy.name}_psbo_comprehensive")
        )
        write_window_report(
            os.path.join(sc_dir, f"{ref_noisy.name}_PSBO_window_report.txt"),
            header_lines=[
                "DATASET: STANFORD (CSV)",
                f"ROBUSTNESS TEST: {sc['name']} - {sc['label']}",
                f"NOISE TYPE: {sc['type']}",
                f"NOISE PARAMETERS: {sc['params']}",
                f"CLEAN-DATA WINDOW: {clean_lo}-{clean_hi}",
            ],
            best=best_n,
            analysis=an_n,
            cell=ref_noisy,
            extra_lines=[
                f"WINDOW SHIFT vs CLEAN: lower {nlo - clean_lo:+d}, upper {nhi - clean_hi:+d}"
            ],
        )
        rows.append(
            {
                "scenario": sc["name"],
                "label": sc["label"],
                "type": sc["type"],
                "params": str(sc["params"]),
                "clean_lower": clean_lo,
                "clean_upper": clean_hi,
                "noisy_lower": nlo,
                "noisy_upper": nhi,
                "lower_shift": nlo - clean_lo,
                "upper_shift": nhi - clean_hi,
                "noisy_score": float(best_n["score"]),
                "rows_clean_window": n_rows_a,
                "rows_noisy_window": n_rows_b,
            }
        )
        print(f"\n  Window shift: lower {nlo - clean_lo:+d}, upper {nhi - clean_hi:+d}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "robustness_summary.csv"), index=False)
    print(f"\n[OK] -> {out}")


def run_selected(choices):
    available = {
        1: option_1_psbo,
        2: option_2_fixed_windows,
        3: option_3_sampling_frequency,
        4: option_4_robustness,
    }
    choices = [int(choice) for choice in choices]
    invalid = set(choices) - available.keys()
    if invalid:
        raise ValueError(
            f"Unsupported options: {sorted(invalid)}; choose {sorted(available)}"
        )
    result = None
    for choice in choices:
        if choice == 1:
            result = available[choice]()
        elif choice in (3, 4):
            if result is None:
                ref_name, cell_ref, cap_map_ref = _reference_setup()
                best, _, _, _ = psbo_optimize(cell_ref, cap_map_ref)
                result = {
                    "best": best,
                    "lower": int(best["lower_soc"]),
                    "upper": int(best["upper_soc"]),
                    "cell_ref": cell_ref,
                    "cap_map_ref": cap_map_ref,
                }
            available[choice](psbo_result=result)
        else:
            available[choice]()


def main(argv=None):
    run_dataset_cli(globals(), run_selected, [1, 2, 3, 4], argv)


if __name__ == "__main__":
    main()
