"""calce_lco_psbo: native dataset processing and retained experiments."""

import os
import pickle
import numpy as np
from scipy import stats
from tqdm import tqdm

if __package__:
    from .psbo_core import (
        safe_mkdir,
        get_memory_mb,
        calc_7_features,
        slice_soc_window,
        psbo_optimize,
        plot_2x2,
        save_figure_3formats,
        extract_features_csv,
        write_window_report,
        assert_reference_held_out,
        FEATURE_NAMES_OUT,
    )
    from .psbo_core import run_dataset_cli
else:
    from psbo_core import (
        safe_mkdir,
        get_memory_mb,
        calc_7_features,
        slice_soc_window,
        psbo_optimize,
        plot_2x2,
        save_figure_3formats,
        extract_features_csv,
        write_window_report,
        assert_reference_held_out,
        FEATURE_NAMES_OUT,
    )
    from psbo_core import run_dataset_cli
CALCE_DIR = "D:\\Feature Extarction\\CALCE\\Data"
OUT_DIR = "D:\\Feature Extarction\\CALCE\\PS-BO Window Selected"
FIXED_WINDOWS = [
    (0, 100, "D:\\Feature Extarction\\CALCE\\0 %- 100% Window Selected"),
    (53, 97, "D:\\Feature Extarction\\CALCE\\NCM 53 %- 97% Window Selected"),
    (15, 63, "D:\\Feature Extarction\\CALCE\\NCA 15 %- 63% Window Selected"),
    (52, 91, "D:\\Feature Extarction\\CALCE\\LFP 52 %- 91% Window Selected"),
    (21, 51, "D:\\Feature Extarction\\CALCE\\NA_ion 21 %- 51% Window Selected"),
    (20, 80, "D:\\Feature Extarction\\CALCE\\20 %- 80% Window Selected"),
    (25, 75, "D:\\Feature Extarction\\CALCE\\25 %- 75% Window Selected"),
    (30, 70, "D:\\Feature Extarction\\CALCE\\30 %- 70% Window Selected"),
    (40, 60, "D:\\Feature Extarction\\CALCE\\40 %- 60% Window Selected"),
]
REFERENCE_CELL = "CALCE_CX2_35.pkl"
CALCE_CELLS = [
    "CALCE_CS2_33.pkl",
    "CALCE_CS2_34.pkl",
    "CALCE_CS2_35.pkl",
    "CALCE_CS2_36.pkl",
    "CALCE_CS2_37.pkl",
    "CALCE_CS2_38.pkl",
    "CALCE_CX2_33.pkl",
    "CALCE_CX2_34.pkl",
    "CALCE_CX2_36.pkl",
    "CALCE_CX2_37.pkl",
    "CALCE_CX2_35.pkl",
]
DATASET_NAME = "CALCE"
LAST_COL_NAME = "Capacity"
CAP_FILTER = np.inf
MODEL_SPLITS = {"train": [], "val": [], "test": ["CALCE_CS2_34", "CALCE_CX2_34"]}
ENFORCE_LEAKAGE_GUARD = True
NATIVE_SOC_MODE = "per_cycle"


class CALCEBattery:
    """
    CALCE .pkl produced by the CALCE preprocessor:
      - cycle_data: list of cycles
      - nominal_capacity_in_Ah from pkl
      - SOC_interval: [0, 1]
    """

    def __init__(self, pkl_path: str):
        self.path = pkl_path
        self.battery_name = os.path.basename(pkl_path).replace(".pkl", "")
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.cycle_data = self.data["cycle_data"]
        self.cycle_life = len(self.cycle_data)
        self.nominal_capacity = float(self.data.get("nominal_capacity_in_Ah", np.nan))
        if not np.isfinite(self.nominal_capacity) or self.nominal_capacity <= 0:
            self.nominal_capacity = 1.1
        soc_interval = self.data.get("SOC_interval", [0, 1])
        try:
            self.soc_interval_width = float(soc_interval[1] - soc_interval[0])
        except Exception:
            self.soc_interval_width = 1.0
        if self.soc_interval_width <= 0:
            self.soc_interval_width = 1.0

    @staticmethod
    def _get_field(cyc, key):
        if isinstance(cyc, dict):
            return cyc.get(key, None)
        return getattr(cyc, key, None)

    def get_capacity_trajectory(self) -> np.ndarray:
        Qd = []
        for cyc in self.cycle_data:
            dc = self._get_field(cyc, "discharge_capacity_in_Ah")
            try:
                Qd.append(float(np.nanmax(np.asarray(dc, dtype=float))))
            except Exception:
                Qd.append(np.nan)
        return np.asarray(Qd, dtype=float)

    def get_cycle_arrays_charge_segment(self, cycle_number_1based: int):
        if cycle_number_1based < 1 or cycle_number_1based > self.cycle_life:
            return None
        cyc = self.cycle_data[cycle_number_1based - 1]
        I = self._get_field(cyc, "current_in_A")
        V = self._get_field(cyc, "voltage_in_V")
        Qc = self._get_field(cyc, "charge_capacity_in_Ah")
        t = self._get_field(cyc, "time_in_s")
        try:
            I = np.asarray(I, dtype=float)
            V = np.asarray(V, dtype=float)
            Qc = np.asarray(Qc, dtype=float)
            t = np.asarray(t, dtype=float)
        except Exception:
            return None
        if len(V) == 0 or len(t) == 0:
            return None
        mask = (
            np.isfinite(I) & (I > 0) & np.isfinite(V) & np.isfinite(Qc) & np.isfinite(t)
        )
        if np.sum(mask) < 3:
            return None
        V = V[mask]
        Qc = Qc[mask]
        t = t[mask]
        t = t - t[0]
        Qc_rel = Qc - Qc[0]
        denom = Qc_rel[-1] if Qc_rel[-1] > 0 else np.nanmax(Qc_rel)
        if not np.isfinite(denom) or denom <= 0:
            denom = 1.0
        soc = Qc_rel / denom * 100.0
        return {"soc": soc.astype(float), "V": V.astype(float), "t": t.astype(float)}


class CALCECell:
    """Native charge segments and source cycle identifiers."""

    def __init__(self, pkl_path):
        self.batt = CALCEBattery(pkl_path)
        self.name = self.batt.battery_name
        self.soc_mode = NATIVE_SOC_MODE
        self.charge_segments = []
        self.cycle_ids = []
        for cyc in range(1, self.batt.cycle_life + 1):
            arr = self.batt.get_cycle_arrays_charge_segment(cyc)
            if arr is None:
                continue
            self.charge_segments.append(arr)
            self.cycle_ids.append(int(cyc))
        self.n_cycles = len(self.charge_segments)


def raw_cap_map(batt) -> dict:
    """Extraction labels: {cycle: Qd_in_Ah}, 0.0 fallback — exactly the
    original Capacity-column behaviour. Covers cycles 1..cycle_life-1 only."""
    traj = batt.get_capacity_trajectory()
    m = {}
    for cyc in range(1, batt.cycle_life):
        idx = cyc - 1
        m[cyc] = float(traj[idx]) if idx < len(traj) and np.isfinite(traj[idx]) else 0.0
    return m


def soh_cap_map(batt) -> dict:
    """PS-BO labels: SOH = Qd / (nominal * SOC_interval_width) — the original
    Life-Label denominator. Non-finite entries are omitted."""
    traj = batt.get_capacity_trajectory()
    denom = batt.nominal_capacity * batt.soc_interval_width
    m = {}
    for cyc in range(1, batt.cycle_life):
        idx = cyc - 1
        if idx < len(traj) and np.isfinite(traj[idx]):
            m[cyc] = float(traj[idx] / denom)
    return m


def stem(fn):
    return str(fn).replace(".pkl", "")


def cell_path(fn):
    return os.path.join(CALCE_DIR, fn if str(fn).endswith(".pkl") else str(fn) + ".pkl")


def _r_by_feature_lines(cell, cap_map, lo, hi):
    """Per-feature Pearson r vs SOH inside the selected window — reproduces the
    r_by_feature block of the original window report."""
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


def extract_all_cells(lower, upper, out_dir, desc=""):
    """Extract 7 SOC features + Capacity for every CALCE cell. CSV naming and
    column order identical to the original script."""
    safe_mkdir(out_dir)
    n_csv = n_rows = 0
    for fn in tqdm(CALCE_CELLS, desc=desc or f"[{lower}%-{upper}%]"):
        p = cell_path(fn)
        if not os.path.exists(p):
            tqdm.write(f"WARNING: missing file -> {p}")
            continue
        try:
            cell = CALCECell(p)
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
    cell_ref = CALCECell(ref_path)
    cap_ref = soh_cap_map(cell_ref.batt)
    return (cell_ref, cap_ref)


def _add_guide_lines(fig, lo, hi):
    """Restore the original CALCE figure look (blue guide lines to the axes)."""
    ax_hm = fig.axes[0]
    ax_hm.plot([lo + 0.5, lo + 0.5], [hi + 0.5, 0], "b-", linewidth=1.5, zorder=5)
    ax_hm.plot([0, lo + 0.5], [hi + 0.5, hi + 0.5], "b-", linewidth=1.5, zorder=5)


def option_1_psbo():
    print("\n" + "=" * 110)
    print("[1] PS-BO WINDOW SELECTION + FEATURE EXTRACTION")
    print("=" * 110)
    safe_mkdir(OUT_DIR)
    print(f"Memory before reference load: {get_memory_mb():.2f} MB")
    cell_ref, cap_ref = _reference_setup()
    print(f"Memory after reference load : {get_memory_mb():.2f} MB")
    print(f"Reference battery: {cell_ref.name} | cycles: {cell_ref.batt.cycle_life}")
    print(f"Nominal capacity from pkl: {cell_ref.batt.nominal_capacity} Ah\n")
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
            "DATASET: CALCE (PKL)",
            f"NOMINAL_CAPACITY_Ah: {cell_ref.batt.nominal_capacity}",
            f"SOC_INTERVAL_WIDTH: {cell_ref.batt.soc_interval_width}",
            "CAPACITY SOURCE: max(discharge_capacity_in_Ah) per cycle",
            "SOC TYPE: charge segment only (I > 0), soc = (Qc_rel / denom) * 100",
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
    fig = plot_2x2(best, score_map, axes, cell_ref, title_text=f"LCO ({cell_ref.name})")
    _add_guide_lines(fig, lo, hi)
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


def run_selected(choices):
    available = {1: option_1_psbo, 2: option_2_fixed_windows}
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
            available[choice](psbo_result=result)
        else:
            available[choice]()


def main(argv=None):
    run_dataset_cli(globals(), run_selected, [1, 2], argv)


if __name__ == "__main__":
    main()
