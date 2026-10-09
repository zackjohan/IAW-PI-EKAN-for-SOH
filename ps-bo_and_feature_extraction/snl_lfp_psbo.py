"""snl_lfp_psbo: native dataset processing and retained experiments."""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

if __package__:
    from .psbo_core import (
        safe_mkdir,
        get_memory_mb,
        downsample_to_hz,
        calc_7_features,
        slice_soc_window,
        evaluate_window,
        psbo_optimize,
        plot_2x2,
        save_figure_3formats,
        extract_features_csv,
        write_window_report,
        assert_reference_held_out,
        apply_init_error_to_soc,
        apply_random_walk_to_soc,
        apply_spike_noise_to_soc,
        FEATURE_NAMES_OUT,
    )
    from .psbo_core import run_dataset_cli
else:
    from psbo_core import (
        safe_mkdir,
        get_memory_mb,
        downsample_to_hz,
        calc_7_features,
        slice_soc_window,
        evaluate_window,
        psbo_optimize,
        plot_2x2,
        save_figure_3formats,
        extract_features_csv,
        write_window_report,
        assert_reference_held_out,
        apply_init_error_to_soc,
        apply_random_walk_to_soc,
        apply_spike_noise_to_soc,
        FEATURE_NAMES_OUT,
    )
    from psbo_core import run_dataset_cli
SNL_DIR = "D:\\Feature Extarction\\SNL\\Data"
OUT_DIR = "D:\\Feature Extarction\\SNL\\PS-BO Window Selected"
FIXED_WINDOWS = [
    (0, 100, "D:\\Feature Extarction\\SNL\\0 %- 100% Window Selected"),
    (53, 97, "D:\\Feature Extarction\\SNL\\NCM 53 %- 97% Window Selected"),
    (15, 63, "D:\\Feature Extarction\\SNL\\NCA 15 %- 63% Window Selected"),
    (13, 87, "D:\\Feature Extarction\\SNL\\LCO 13 %- 87% Window Selected"),
    (21, 51, "D:\\Feature Extarction\\SNL\\NA_ion 21 %- 51% Window Selected"),
    (20, 80, "D:\\Feature Extarction\\SNL\\20 %- 80% Window Selected"),
    (25, 75, "D:\\Feature Extarction\\SNL\\25 %- 75% Window Selected"),
    (30, 70, "D:\\Feature Extarction\\SNL\\30 %- 70% Window Selected"),
    (40, 60, "D:\\Feature Extarction\\SNL\\40 %- 60% Window Selected"),
]
REFERENCE_CELL = "SNL_18650_LFP_35C_0-100_0.5-2C_a.pkl"
SNL_CELLS = [
    "SNL_18650_LFP_15C_0-100_0.5-2C_b.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-0.5C_a.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-1C_a.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-1C_b.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-2C_a.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-2C_b.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-3C_a.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-3C_b.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-3C_c.pkl",
    "SNL_18650_LFP_25C_0-100_0.5-3C_d.pkl",
    "SNL_18650_LFP_35C_0-100_0.5-1C_a.pkl",
    "SNL_18650_LFP_35C_0-100_0.5-1C_b.pkl",
    "SNL_18650_LFP_35C_0-100_0.5-1C_c.pkl",
    "SNL_18650_LFP_35C_0-100_0.5-1C_d.pkl",
    "SNL_18650_LFP_35C_0-100_0.5-2C_a.pkl",
    "SNL_18650_LFP_35C_0-100_0.5-2C_b.pkl",
]
MODULES_ROOT = "D:\\Feature Extarction\\SNL\\PSBO_Modules"
MODULE_DIRS = {
    "M6_sampling_frequency": os.path.join(MODULES_ROOT, "06_Sampling_Frequency"),
    "M8_robustness": os.path.join(MODULES_ROOT, "08_Robustness"),
}
DATASET_NAME = "SNL_LFP"
LAST_COL_NAME = "Capacity"
CAP_FILTER = np.inf
MODEL_SPLITS = {
    "train": [],
    "val": [],
    "test": [
        "SNL_18650_LFP_25C_0-100_0.5-1C_a",
        "SNL_18650_LFP_25C_0-100_0.5-1C_b",
        "SNL_18650_LFP_35C_0-100_0.5-1C_c",
    ],
}
ENFORCE_LEAKAGE_GUARD = True
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
NATIVE_SOC_MODE = "per_cycle"


class SNLBattery:

    def __init__(self, pkl_path: str):
        self.path = pkl_path
        self.battery_name = os.path.basename(pkl_path).replace(".pkl", "")
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.cycle_data = self.data["cycle_data"]
        self.cycle_life = len(self.cycle_data)
        self.nominal_capacity = float(self.data.get("nominal_capacity_in_Ah", 1.0))
        soc_interval = self.data.get("SOC_interval", [0, 1])
        self.soc_interval_width = (
            float(soc_interval[1] - soc_interval[0]) if len(soc_interval) == 2 else 1.0
        )

    def get_capacity_trajectory(self) -> np.ndarray:
        Qd = []
        for cyc in self.cycle_data:
            try:
                Qd.append(float(np.nanmax(cyc["discharge_capacity_in_Ah"])))
            except Exception:
                Qd.append(np.nan)
        return np.asarray(Qd, dtype=float)

    def get_cycle_arrays(self, cycle_number_1based: int):
        if cycle_number_1based < 1 or cycle_number_1based > self.cycle_life:
            return None
        cyc = self.cycle_data[cycle_number_1based - 1]
        try:
            I = np.asarray(cyc["current_in_A"], dtype=float)
            V = np.asarray(cyc["voltage_in_V"], dtype=float)
            Qc = np.asarray(cyc["charge_capacity_in_Ah"], dtype=float)
            t = np.asarray(cyc["time_in_s"], dtype=float)
        except Exception:
            return None
        mask_chg = (
            np.isfinite(I) & (I > 0) & np.isfinite(V) & np.isfinite(Qc) & np.isfinite(t)
        )
        if np.sum(mask_chg) < 3:
            return None
        I = I[mask_chg]
        V = V[mask_chg]
        Qc = Qc[mask_chg]
        t = t[mask_chg]
        t = t - t[0]
        Qc_rel = Qc - Qc[0]
        max_q = Qc_rel[-1] if Qc_rel[-1] > 0 else np.nanmax(Qc_rel)
        max_q = max_q if max_q > 0 else 1.0
        soc = Qc_rel / max_q * 100.0
        return {"soc": soc.astype(float), "V": V.astype(float), "t": t.astype(float)}

    def get_cycle_arrays_raw(self, cycle_number_1based: int):
        if cycle_number_1based < 1 or cycle_number_1based > self.cycle_life:
            return None
        cyc = self.cycle_data[cycle_number_1based - 1]
        try:
            I = np.asarray(cyc["current_in_A"], dtype=float)
            V = np.asarray(cyc["voltage_in_V"], dtype=float)
            Qc = np.asarray(cyc["charge_capacity_in_Ah"], dtype=float)
            t = np.asarray(cyc["time_in_s"], dtype=float)
        except Exception:
            return None
        mask_chg = (
            np.isfinite(I) & (I > 0) & np.isfinite(V) & np.isfinite(Qc) & np.isfinite(t)
        )
        if np.sum(mask_chg) < 3:
            return None
        V, Qc, t = (V[mask_chg], Qc[mask_chg], t[mask_chg])
        t = t - t[0]
        Qc_rel = Qc - Qc[0]
        max_q = Qc_rel[-1] if Qc_rel[-1] > 0 else np.nanmax(Qc_rel)
        max_q = max_q if max_q > 0 else 1.0
        return {
            "Qc_rel": Qc_rel.astype(float),
            "denom": float(max_q),
            "V": V.astype(float),
            "t": t.astype(float),
        }


def _apply_current_bias_to_soc(soc, t, bias_crate):
    """Scenario A translated for Qc-based SOC (b*t/36 pp, same error model
    as the Stanford implementation). Not clipped."""
    return soc + float(bias_crate) * t / 36.0


class SNLCell:

    def __init__(self, pkl_path, noise_config=None, noise_seed=42, target_hz=None):
        self.batt = SNLBattery(pkl_path)
        self.name = self.batt.battery_name
        self.soc_mode, self.target_hz = (NATIVE_SOC_MODE, target_hz)
        self.noise_config, self.noise_seed = (noise_config, int(noise_seed))
        self._cap_traj = self.batt.get_capacity_trajectory()
        self.charge_segments = []
        self.cycle_ids = []
        for cyc in range(1, self.batt.cycle_life + 1):
            plain = (
                noise_config is None
                and target_hz is None
                and (NATIVE_SOC_MODE == NATIVE_SOC_MODE)
            )
            arr = (
                self.batt.get_cycle_arrays(cyc)
                if plain
                else self._segment_with_options(cyc)
            )
            if arr is None:
                continue
            self.charge_segments.append(arr)
            self.cycle_ids.append(int(cyc))
        self.n_cycles = len(self.charge_segments)

    def _segment_with_options(self, cyc):
        raw = self.batt.get_cycle_arrays_raw(cyc)
        if raw is None:
            return None
        Qc_rel, denom = (raw["Qc_rel"], raw["denom"])
        V, t = (raw["V"], raw["t"])
        soc = Qc_rel / denom * 100.0
        if self.noise_config is not None:
            cfg = self.noise_config
            if cfg["type"] == "bias_drift":
                soc = _apply_current_bias_to_soc(soc, t, cfg["params"]["bias_crate"])
            else:
                local_seed = int(
                    self.noise_seed
                    + int(cyc) * 1000
                    + sum((ord(ch) for ch in self.name))
                )
                rng = np.random.default_rng(local_seed)
                p = cfg["params"]
                if cfg["type"] == "init_error":
                    soc = apply_init_error_to_soc(soc, p["init_error_percent"])
                elif cfg["type"] == "random_walk":
                    soc = apply_random_walk_to_soc(soc, p["sigma_step"], rng)
                elif cfg["type"] == "spike_noise":
                    soc = apply_spike_noise_to_soc(
                        soc, p["spike_prob"], p["spike_mag"], rng
                    )
        if self.target_hz is not None:
            keep = downsample_to_hz(t, target_hz=self.target_hz)
            soc, V, t = (soc[keep], V[keep], t[keep])
            if len(t) == 0:
                return None
        return {
            "soc": np.asarray(soc, dtype=float),
            "V": np.asarray(V, dtype=float),
            "t": np.asarray(t, dtype=float),
        }


def raw_cap_map(batt) -> dict:
    traj = batt.get_capacity_trajectory()
    m = {}
    for cyc in range(1, batt.cycle_life):
        idx = cyc - 1
        m[cyc] = float(traj[idx]) if idx < len(traj) and np.isfinite(traj[idx]) else 0.0
    return m


def soh_cap_map(batt) -> dict:
    traj = batt.get_capacity_trajectory()
    denom = batt.nominal_capacity * (
        batt.soc_interval_width if batt.soc_interval_width > 0 else 1.0
    )
    m = {}
    for cyc in range(1, batt.cycle_life):
        idx = cyc - 1
        if idx < len(traj) and np.isfinite(traj[idx]):
            m[cyc] = float(traj[idx] / denom)
    return m


def stem(fn):
    return str(fn).replace(".pkl", "")


def cell_path(fn):
    return os.path.join(SNL_DIR, fn if str(fn).endswith(".pkl") else str(fn) + ".pkl")


def build_cell(fn, **kw):
    return SNLCell(cell_path(fn), **kw)


def _r_by_feature_lines(cell, cap_map, lo, hi):
    y = np.full(cell.n_cycles, np.nan)
    for i, cid in enumerate(cell.cycle_ids):
        if cid in cap_map:
            y[i] = cap_map[cid]
    lines = []
    for name in FEATURE_NAMES_OUT:
        x = np.full(cell.n_cycles, np.nan)
        for i in range(cell.n_cycles):
            if not np.isfinite(y[i]):
                continue
            w = slice_soc_window(cell.charge_segments[i], lo, hi, min_points=2)
            if w is None:
                continue
            f = calc_7_features(w)
            if f is None:
                continue
            x[i] = f.get(name, np.nan)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) < 2:
            continue
        try:
            r, _ = stats.pearsonr(x[mask], y[mask])
            lines.append(f"  {name}: {float(r):.8f}")
        except Exception:
            continue
    return lines


def extract_all_cells(
    lower, upper, out_dir, noise_config=None, target_hz=None, desc=""
):
    safe_mkdir(out_dir)
    n_csv = n_rows = 0
    for fn in tqdm(SNL_CELLS, desc=desc or f"[{lower}%-{upper}%]"):
        p = cell_path(fn)
        if not os.path.exists(p):
            tqdm.write(f"WARNING: missing file -> {p}")
            continue
        try:
            cell = SNLCell(
                p,
                noise_config=noise_config,
                noise_seed=SCENARIO_RANDOM_SEED,
                target_hz=target_hz,
            )
            df_feat = extract_features_csv(
                cell,
                raw_cap_map(cell.batt),
                lower,
                upper,
                diag_interp=None,
                cap_threshold=CAP_FILTER,
                last_col_name=LAST_COL_NAME,
            )
            out_csv = os.path.join(out_dir, stem(fn) + ".csv")
            df_feat.to_csv(out_csv, index=False)
            n_csv += 1
            n_rows += len(df_feat)
            tqdm.write(f"  {stem(fn)}.csv: saved {len(df_feat)} rows -> {out_csv}")
        except Exception as e:
            tqdm.write(f"ERROR: {fn} -> {e}")
    return (n_csv, n_rows)


def _reference_setup():
    ref_path = cell_path(REFERENCE_CELL)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference cell not found: {ref_path}")
    if ENFORCE_LEAKAGE_GUARD:
        assert_reference_held_out(REFERENCE_CELL, MODEL_SPLITS, DATASET_NAME)
    cell_ref = SNLCell(ref_path)
    cap_ref = soh_cap_map(cell_ref.batt)
    return (cell_ref, cap_ref)


def option_1_psbo():
    print("\n" + "=" * 110)
    print("[1] PS-BO WINDOW SELECTION + FEATURE EXTRACTION")
    print("=" * 110)
    safe_mkdir(OUT_DIR)
    print(f"Memory before reference load: {get_memory_mb():.2f} MB")
    cell_ref, cap_ref = _reference_setup()
    print(f"Memory after reference load : {get_memory_mb():.2f} MB")
    print(f"Reference battery: {cell_ref.name} | cycles: {cell_ref.batt.cycle_life}\n")
    best, score_map, axes, analysis = psbo_optimize(cell_ref, cap_ref)
    lo, hi = (int(best["lower_soc"]), int(best["upper_soc"]))
    print("\n" + "=" * 90)
    print("PS-BO SUMMARY")
    print("=" * 90)
    print(f"BEST WINDOW: lower={lo} upper={hi}")
    print(f"BEST SCORE (mean |r|): {best['score']:.6f}")
    print(f"WINDOW COVERAGE: {best.get('coverage', 0.0) * 100:.2f}%")
    print(f"Optimization time: {analysis.get('search_time_sec', 0.0):.2f} sec")
    write_window_report(
        os.path.join(OUT_DIR, f"{cell_ref.name}_PSBO_window_report.txt"),
        header_lines=[
            "DATASET: SNL LFP (PKL)  — NCA/NMC groups removed (as in original)",
            f"NOMINAL_CAPACITY_Ah: {cell_ref.batt.nominal_capacity}",
            f"SOC_INTERVAL_WIDTH: {cell_ref.batt.soc_interval_width}",
            "CAPACITY SOURCE: max(discharge_capacity_in_Ah) per cycle",
            "SOC TYPE: charge segment only (I > 0), soc = (Qc_rel / max_q) * 100",
            "PS-BO LABEL: SOH = Qd / (nominal * SOC_interval_width)",
            "EXTRACTION RANGE: cycles 1..cycle_life-1 (original convention)",
            "NOTE: BV_Driver_T and Heating_I2 removed (7 SOC features only)",
        ],
        best=best,
        analysis=analysis,
        cell=cell_ref,
        extra_lines=["r_by_feature (SOH labels):"]
        + _r_by_feature_lines(cell_ref, cap_ref, lo, hi),
    )
    fig = plot_2x2(best, score_map, axes, cell_ref, title_text=f"LFP ({cell_ref.name})")
    save_figure_3formats(
        fig, os.path.join(OUT_DIR, f"{cell_ref.name}_psbo_comprehensive")
    )
    print("\n[OK] Saved 2x2 figure (PNG/PDF/SVG)\n")
    extract_all_cells(lo, hi, OUT_DIR, desc=f"PS-BO window [{lo}%-{hi}%]")
    print(f"\n[OK] PS-BO extraction complete -> {OUT_DIR}")
    print(f"     Selected SOC window: {lo}%-{hi}%")
    return {
        "best": best,
        "lower": lo,
        "upper": hi,
        "cell_ref": cell_ref,
        "cap_map_ref": cap_ref,
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
        cell_ref, cap_ref = _reference_setup()
        best, _, _, _ = psbo_optimize(cell_ref, cap_ref)
        lo, hi = (int(best["lower_soc"]), int(best["upper_soc"]))
    else:
        lo, hi = (psbo_result["lower"], psbo_result["upper"])
    print(f"Window held fixed at the PS-BO selection: {lo}%-{hi}%")
    print("NOTE: the native SNL pipeline uses the pkl's full time resolution")
    print("(options 1/2). Every rate below decimates those SAME native segments.\n")
    rows = []
    for hz in SAMPLING_SWEEP_HZ:
        d = os.path.join(out, f"{hz}Hz")
        print(f"\n  --- {hz} Hz  ->  {d} ---")
        n_csv, n_rows_w = extract_all_cells(lo, hi, d, target_hz=hz, desc=f"{hz} Hz")
        try:
            ref = build_cell(REFERENCE_CELL, target_hz=hz)
            res = evaluate_window(lo, hi, ref, soh_cap_map(ref.batt))
        except Exception:
            res = None
        rows.append(
            {
                "target_hz": hz,
                "lower_soc": lo,
                "upper_soc": hi,
                "csv_written": n_csv,
                "rows_written": n_rows_w,
                "window_score": res["score"] if res else np.nan,
                "coverage": res["coverage"] if res else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(out, "sampling_frequency_summary.csv"), index=False
    )
    print(f"\n[OK] -> {out}")


def option_4_robustness(psbo_result=None):
    print("\n" + "=" * 110)
    print("[4] ROBUSTNESS SCENARIOS A-D")
    print("=" * 110)
    out = MODULE_DIRS["M8_robustness"]
    safe_mkdir(out)
    if psbo_result is None:
        cell_ref, cap_ref = _reference_setup()
        best, _, _, _ = psbo_optimize(cell_ref, cap_ref)
        clean_lo, clean_hi = (int(best["lower_soc"]), int(best["upper_soc"]))
    else:
        cap_ref = psbo_result["cap_map_ref"]
        clean_lo, clean_hi = (psbo_result["lower"], psbo_result["upper"])
    print(f"Clean-data PS-BO window: {clean_lo}%-{clean_hi}%\n")
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
        try:
            ref_noisy = build_cell(
                REFERENCE_CELL, noise_config=sc, noise_seed=SCENARIO_RANDOM_SEED
            )
            best_n, score_map_n, axes_n, an_n = psbo_optimize(
                ref_noisy, cap_ref, desc_suffix=f" [{sc['name']}]"
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
                title_text=f"LFP ({ref_noisy.name}) — {sc['name']}: {sc['label']}",
            )
            save_figure_3formats(
                fig, os.path.join(sc_dir, f"{ref_noisy.name}_psbo_comprehensive")
            )
            write_window_report(
                os.path.join(sc_dir, f"{ref_noisy.name}_PSBO_window_report.txt"),
                header_lines=[
                    "DATASET: SNL LFP (PKL)",
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
            print(
                f"\n  Window shift: lower {nlo - clean_lo:+d}, upper {nhi - clean_hi:+d}"
            )
            rec = {
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
        except Exception as e:
            print(f"\n  ERROR re-selecting window for {sc['name']}: {e}")
            rec = {
                "scenario": sc["name"],
                "label": sc["label"],
                "type": sc["type"],
                "params": str(sc["params"]),
                "clean_lower": clean_lo,
                "clean_upper": clean_hi,
                "noisy_lower": np.nan,
                "noisy_upper": np.nan,
                "lower_shift": np.nan,
                "upper_shift": np.nan,
                "noisy_score": np.nan,
                "rows_clean_window": n_rows_a,
                "rows_noisy_window": np.nan,
            }
        rows.append(rec)
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
                cell_ref, cap_map_ref = _reference_setup()
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
