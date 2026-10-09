"""xjtu_ncm_psbo: native dataset processing and retained experiments."""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib import rcParams
from tqdm import tqdm
import psutil
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

if __package__:
    from .psbo_core import run_dataset_cli
else:
    from psbo_core import run_dataset_cli
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 10
BATCH_DIRS = [
    Path("D:\\Feature Extarction\\XJTU\\Data\\Batch-1"),
    Path("D:\\Feature Extarction\\XJTU\\Data\\Batch-2"),
    Path("D:\\Feature Extarction\\XJTU\\Data\\Batch-3"),
]
PSBO_ROOT = Path("D:\\Feature Extarction\\XJTU\\PS-BO Window Selected")
FIXED_WINDOWS = [
    (0, 100, Path("D:\\Feature Extarction\\XJTU\\0 %- 100% Window Selected")),
    (15, 63, Path("D:\\Feature Extarction\\XJTU\\NCA 15 %- 63% Window Selected")),
    (52, 91, Path("D:\\Feature Extarction\\XJTU\\LFP 52 %- 91% Window Selected")),
    (13, 87, Path("D:\\Feature Extarction\\XJTU\\LCO 13 %- 87% Window Selected")),
    (21, 51, Path("D:\\Feature Extarction\\XJTU\\NA_ion 21 %- 51% Window Selected")),
    (20, 80, Path("D:\\Feature Extarction\\XJTU\\20 %- 80% Window Selected")),
    (25, 75, Path("D:\\Feature Extarction\\XJTU\\25 %- 75% Window Selected")),
    (30, 70, Path("D:\\Feature Extarction\\XJTU\\30 %- 70% Window Selected")),
    (40, 60, Path("D:\\Feature Extarction\\XJTU\\40 %- 60% Window Selected")),
]
REFERENCE_FILE = "R2.5_battery-3.mat"
NOMINAL_CAPACITY_AH = 2.0
LAST_COL_NAME = "Capacity"
SOC_FEATURE_NAMES = [
    "ChargingTime",
    "VoltageSlope",
    "VoltageStd",
    "VoltageKurtosis",
    "VoltageSkewness",
    "VoltageEntropy",
    "VoltageCurvature",
]
FEATURE_NAMES_OUT = SOC_FEATURE_NAMES
FEATURE_NAMES = SOC_FEATURE_NAMES
MIN_POINTS_GRID = 2
MIN_POINTS_FEATURES = 10
MIN_CYCLES_GRID = 2
MIN_FEATURES_OK = 6
USE_ABS_CORR = True
SOC_MIN = 0
SOC_MAX = 100
PSBO_N_CALLS = 120
PSBO_N_INITIAL = 25
PSBO_RANDOM_STATE = 42


def get_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def safe_load_xjtu_mat(path: str) -> dict:
    """
    Load a XJTU .mat file.  Tries scipy.loadmat first; falls back to h5py
    for MATLAB v7.3 / HDF5 files.
    """
    try:
        return loadmat(path, variable_names=["data", "summary"])
    except Exception as e1:
        try:
            import h5py

            mat = {}
            with h5py.File(path, "r") as f:
                for key in ["data", "summary"]:
                    if key in f:
                        mat[key] = f[key][()]
                    else:
                        raise KeyError(f"Key '{key}' not found in v7.3 mat file.")
            return mat
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load MAT file.\nscipy.loadmat error: {e1}\nh5py fallback error: {e2}\nFile: {path}"
            )


class XJTUBattery:
    """
    Loads a XJTU NCM .mat file and builds per-cycle caches used by:
      • PS-BO window selection  (SOC features only)
    """

    def __init__(self, path: str):
        self.path = path
        self.battery_name = os.path.basename(path).split(".")[0]
        mat = safe_load_xjtu_mat(path)
        self.data = mat["data"]
        self.summary = mat["summary"]
        self.cycle_life = int(self.summary[0][0][8][0][0])
        self.variable_map = {"time": 1, "voltage": 2, "current": 3, "capacity": 4}
        self.batch_num = self._determine_batch()
        self.cap_traj = self.get_capacity_trajectory()
        self.cycle_cache: dict = {}
        self._build_cycle_cache()

    def _determine_batch(self) -> int:
        path_lower = self.path.lower()
        for i in range(1, 7):
            if f"batch-{i}" in path_lower or f"batch{i}" in path_lower:
                return i
        return 1

    def get_one_cycle_description(self, cycle: int) -> str:
        if cycle > self.data.shape[1] or cycle < 1:
            return ""
        try:
            return str(self.data[0][cycle - 1][7][0])
        except Exception:
            return ""

    def get_capacity_trajectory(self):
        raw_capacity = self.summary[0][0][1].reshape(-1)
        if self.batch_num <= 3:
            return raw_capacity
        return self._get_degradation_trajectory_interpolated(raw_capacity)

    def _get_degradation_trajectory_interpolated(self, raw_capacity):
        try:
            test_cycles = []
            for i in range(1, self.cycle_life + 1):
                desc = self.get_one_cycle_description(i)
                if "test capacity" in desc.lower():
                    test_cycles.append(i)
            if len(test_cycles) == 0:
                return raw_capacity
            test_indices = np.array(test_cycles) - 1
            test_capacity = raw_capacity[test_indices]
            all_cycles = np.arange(1, self.cycle_life + 1)
            try:
                f = interpolate.interp1d(
                    test_cycles, test_capacity, kind="cubic", fill_value="extrapolate"
                )
            except Exception:
                f = interpolate.interp1d(
                    test_cycles, test_capacity, kind="linear", fill_value="extrapolate"
                )
            return f(all_cycles)
        except Exception:
            return raw_capacity

    def get_cycle_data(self, cycle: int):
        if cycle > self.data.shape[1] or cycle < 1:
            return None
        try:
            d = self.data[0][cycle - 1]
            t = d[self.variable_map["time"]].reshape(-1)
            v = d[self.variable_map["voltage"]].reshape(-1)
            i = d[self.variable_map["current"]].reshape(-1)
            cap = d[self.variable_map["capacity"]].reshape(-1)
            boundaries = np.where(t == 0)[0]
            if len(boundaries) > 0:
                end = boundaries[1] if len(boundaries) > 1 else len(t)
                return {
                    "time": t[:end],
                    "voltage": v[:end],
                    "current": i[:end],
                    "capacity": cap[:end],
                }
            return {"time": t, "voltage": v, "current": i, "capacity": cap}
        except Exception:
            return None

    def _build_cycle_cache(self):
        for cyc in range(1, self.cycle_life + 1):
            cd = self.get_cycle_data(cyc)
            if cd is None:
                self.cycle_cache[cyc] = {"valid": False}
                continue
            t = np.asarray(cd["time"], dtype=float)
            v = np.asarray(cd["voltage"], dtype=float)
            cap = np.asarray(cd["capacity"], dtype=float)
            m = np.isfinite(t) & np.isfinite(v) & np.isfinite(cap)
            t, v, cap = (t[m], v[m], cap[m])
            if len(t) < MIN_POINTS_FEATURES:
                self.cycle_cache[cyc] = {"valid": False}
                continue
            order = np.argsort(t)
            t, v, cap = (t[order], v[order], cap[order])
            t_sec = (t - t[0]) * 60.0
            cap_norm = cap - cap[0]
            max_cap = cap_norm[-1] if cap_norm[-1] > 0 else 1.0
            soc = cap_norm / max_cap * 100.0
            idx0 = cyc - 1
            y_cap = float(self.cap_traj[idx0]) if idx0 < len(self.cap_traj) else np.nan
            V = v.astype(float)
            tt = t_sec.astype(float)
            N = len(V)
            c1 = np.zeros(N + 1)
            c1[1:] = np.cumsum(V)
            c2 = np.zeros(N + 1)
            c2[1:] = np.cumsum(V**2)
            c3 = np.zeros(N + 1)
            c3[1:] = np.cumsum(V**3)
            c4 = np.zeros(N + 1)
            c4[1:] = np.cumsum(V**4)
            ct = np.zeros(N + 1)
            ct[1:] = np.cumsum(tt)
            ct2 = np.zeros(N + 1)
            ct2[1:] = np.cumsum(tt**2)
            ctv = np.zeros(N + 1)
            ctv[1:] = np.cumsum(tt * V)
            self.cycle_cache[cyc] = {
                "valid": True,
                "y_cap": y_cap,
                "V": V,
                "t": tt,
                "soc": soc.astype(float),
                "N": N,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "ct": ct,
                "ct2": ct2,
                "ctv": ctv,
            }


def _window_indices_from_soc(soc, lower, upper):
    idx = np.where((soc >= lower) & (soc <= upper))[0]
    if len(idx) == 0:
        return (None, None)
    return (int(idx[0]), int(idx[-1] + 1))


def _entropy_from_values(V):
    N = len(V)
    if N <= 0:
        return 0.0
    B = min(100, max(1, N // 10))
    hist, _ = np.histogram(V, bins=B)
    pk = hist / max(N, 1)
    pk = pk[pk > 0]
    return float(-np.sum(pk * np.log2(pk))) if len(pk) else 0.0


def _calc_features_fast_from_ab(d: dict, a: int, b: int):
    N = b - a
    if N < MIN_POINTS_FEATURES:
        return None
    V = d["V"]
    t = d["t"]
    c1 = d["c1"]
    c2 = d["c2"]
    c3 = d["c3"]
    c4 = d["c4"]
    ct = d["ct"]
    ct2 = d["ct2"]
    ctv = d["ctv"]
    s1 = c1[b] - c1[a]
    s2 = c2[b] - c2[a]
    s3 = c3[b] - c3[a]
    s4 = c4[b] - c4[a]
    st = ct[b] - ct[a]
    st2 = ct2[b] - ct2[a]
    stv = ctv[b] - ctv[a]
    feats = {}
    feats["ChargingTime"] = float(t[b - 1] - t[a])
    denom = N * st2 - st**2
    feats["VoltageSlope"] = (
        float((N * stv - st * s1) / denom) if abs(denom) > 1e-12 else 0.0
    )
    mean = s1 / N
    var = s2 / N - mean * mean
    if N > 1:
        var *= N / (N - 1)
    var = max(var, 0.0)
    feats["VoltageStd"] = float(np.sqrt(var))
    m2 = s2 / N - mean**2
    m3 = s3 / N - 3 * mean * (s2 / N) + 2 * mean**3
    m4 = s4 / N - 4 * mean * (s3 / N) + 6 * mean**2 * (s2 / N) - 3 * mean**4
    if m2 > 1e-18:
        feats["VoltageSkewness"] = float(m3 / m2**1.5)
        feats["VoltageKurtosis"] = float(m4 / m2**2 - 3.0)
    else:
        feats["VoltageSkewness"] = 0.0
        feats["VoltageKurtosis"] = 0.0
    feats["VoltageEntropy"] = _entropy_from_values(V[a:b])
    try:
        tt = t[a:b]
        vv = V[a:b]
        X = np.column_stack([tt**2, tt, np.ones(N)])
        coeffs = np.linalg.lstsq(X, vv, rcond=None)[0]
        feats["VoltageCurvature"] = float(coeffs[0])
    except Exception:
        feats["VoltageCurvature"] = 0.0
    return feats


def evaluate_window_score_fast(
    batt: XJTUBattery, lower: int, upper: int, nominal_capacity_ah: float
):
    cycles = np.arange(1, batt.cycle_life + 1)
    y = np.full(len(cycles), np.nan, dtype=float)
    x_by_feat = {f: np.full(len(cycles), np.nan, dtype=float) for f in FEATURE_NAMES}
    for i, cyc in enumerate(cycles):
        d = batt.cycle_cache.get(int(cyc), None)
        if d is None or not d.get("valid", False):
            continue
        cap = d.get("y_cap", np.nan)
        if np.isfinite(cap):
            y[i] = float(cap) / float(nominal_capacity_ah)
        else:
            y[i] = np.nan
        a, b = _window_indices_from_soc(d["soc"], lower, upper)
        if a is None or b - a < MIN_POINTS_GRID:
            continue
        feats = _calc_features_fast_from_ab(d, a, b)
        if feats is None:
            continue
        for f in FEATURE_NAMES:
            x_by_feat[f][i] = feats.get(f, np.nan)
    r_by_feature, p_by_feature, n_by_feature = ({}, {}, {})
    for f in FEATURE_NAMES:
        x = x_by_feat[f]
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(np.sum(mask))
        n_by_feature[f] = n
        if n < MIN_CYCLES_GRID:
            continue
        try:
            r, p = stats.pearsonr(x[mask], y[mask])
            if np.isfinite(r):
                r_by_feature[f] = float(r)
                p_by_feature[f] = float(p)
        except Exception:
            continue
    if len(r_by_feature) == 0 or len(r_by_feature) < MIN_FEATURES_OK:
        return None
    rs = np.array(list(r_by_feature.values()), dtype=float)
    score = float(np.mean(np.abs(rs))) if USE_ABS_CORR else float(np.mean(rs))
    return {
        "lower_soc": int(lower),
        "upper_soc": int(upper),
        "score": float(score),
        "r_by_feature": r_by_feature,
        "p_by_feature": p_by_feature,
        "n_cycles_by_feature": n_by_feature,
    }


def psbo_select_window_fast(
    batt: XJTUBattery, nominal_capacity_ah: float, show_progress: bool = True
):
    score_map = np.full((101, 101), np.nan, dtype=float)
    best = {
        "lower_soc": None,
        "upper_soc": None,
        "score": -np.inf,
        "r_by_feature": None,
        "p_by_feature": None,
        "n_cycles_by_feature": None,
    }
    space = [
        Integer(SOC_MIN, SOC_MAX, name="lower_soc"),
        Integer(SOC_MIN, SOC_MAX, name="upper_soc"),
    ]
    pbar = tqdm(
        total=PSBO_N_CALLS,
        desc=f"PS-BO search [{batt.battery_name}]",
        disable=not show_progress,
    )
    t0 = time.perf_counter()
    mem0 = get_memory_mb()

    @use_named_args(space)
    def objective(lower_soc, upper_soc):
        pbar.update(1)
        if upper_soc < lower_soc:
            return 1.0
        res = evaluate_window_score_fast(
            batt, lower_soc, upper_soc, nominal_capacity_ah
        )
        if res is None:
            score_map[upper_soc, lower_soc] = np.nan
            return 1.0
        score_map[upper_soc, lower_soc] = res["score"]
        if res["score"] > best["score"]:
            best.update(res)
        return -res["score"]

    gp_minimize(
        objective,
        space,
        n_calls=PSBO_N_CALLS,
        n_initial_points=PSBO_N_INITIAL,
        acq_func="EI",
        random_state=PSBO_RANDOM_STATE,
    )
    pbar.close()
    t1 = time.perf_counter()
    mem1 = get_memory_mb()
    if best["lower_soc"] is None:
        for l in range(0, 101, 5):
            for u in range(l, 101, 5):
                res = evaluate_window_score_fast(batt, l, u, nominal_capacity_ah)
                if res is None:
                    continue
                score_map[u, l] = res["score"]
                if res["score"] > best["score"]:
                    best.update(res)
    if best["lower_soc"] is None:
        raise RuntimeError(f"No valid SOC window found for {batt.battery_name}")
    analysis = {
        "search_time_sec": float(t1 - t0),
        "memory_before_mb": float(mem0),
        "memory_after_mb": float(mem1),
        "memory_delta_mb": float(mem1 - mem0),
        "total_windows": int(PSBO_N_CALLS),
        "method": "PS-BO",
    }
    return (best, score_map, (np.arange(0, 101), np.arange(0, 101)), analysis)


def save_figure_csvs(
    batt: XJTUBattery, best: dict, score_map: np.ndarray, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    lower_soc = int(best["lower_soc"])
    upper_soc = int(best["upper_soc"])
    rows_a = []
    for u in range(101):
        for l in range(101):
            rows_a.append(
                {
                    "lower_soc": l,
                    "upper_soc": u,
                    "score": score_map[u, l],
                    "invalid_region_upper_lt_lower": int(u < l),
                    "best_lower_soc": lower_soc,
                    "best_upper_soc": upper_soc,
                }
            )
    pd.DataFrame(rows_a).to_csv(output_dir / "Figure (a).csv", index=False)
    rows_b, rows_c, rows_d = ([], [], [])
    for cycle in range(1, batt.cycle_life + 1):
        d = batt.cycle_cache.get(cycle, None)
        if d is None or not d.get("valid", False):
            continue
        soc = d["soc"]
        mask = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(mask) == 0:
            continue
        t_sec = d["t"][mask]
        v = d["V"][mask]
        s = soc[mask]
        for i in range(len(t_sec)):
            rows_b.append(
                {"cycle": cycle, "time_s": float(t_sec[i]), "soc": float(s[i])}
            )
            rows_c.append(
                {"cycle": cycle, "voltage_v": float(v[i]), "soc": float(s[i])}
            )
            rows_d.append(
                {"cycle": cycle, "time_s": float(t_sec[i]), "voltage_v": float(v[i])}
            )
    pd.DataFrame(rows_b).to_csv(output_dir / "Figure (b).csv", index=False)
    pd.DataFrame(rows_c).to_csv(output_dir / "Figure (c).csv", index=False)
    pd.DataFrame(rows_d).to_csv(output_dir / "Figure (d).csv", index=False)


def plot_comprehensive_visualization(
    best: dict, score_map: np.ndarray, axes: tuple, batt: XJTUBattery
):
    lower_vals, upper_vals = axes
    lower_soc = int(best["lower_soc"])
    upper_soc = int(best["upper_soc"])
    fig = plt.figure(figsize=(6, 5.5))
    fig.suptitle("NCM (R2.5_battery-3)", fontsize=12, y=1.05)
    gs = fig.add_gridspec(
        2, 2, hspace=0.45, wspace=0.4, left=0.08, right=0.95, top=0.95, bottom=0.08
    )
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_soc_time = fig.add_subplot(gs[0, 1])
    ax_soc_voltage = fig.add_subplot(gs[1, 0])
    ax_voltage_time = fig.add_subplot(gs[1, 1])
    score_map_full = np.full((101, 101), np.nan, dtype=float)
    for upper in upper_vals:
        for lower in lower_vals:
            score_map_full[int(upper), int(lower)] = score_map[int(upper), int(lower)]
    mask_full = np.zeros_like(score_map_full, dtype=bool)
    for upper in range(101):
        for lower in range(101):
            if upper < lower:
                mask_full[upper, lower] = True
    sns.heatmap(
        score_map_full,
        mask=mask_full,
        cmap="RdBu_r",
        cbar_kws={"label": "Aggregate Score", "shrink": 0.8},
        ax=ax_heatmap,
        xticklabels=False,
        yticklabels=False,
    )
    tick_vals = list(range(0, 101, 20))
    ax_heatmap.set_xticks([v + 0.5 for v in tick_vals])
    ax_heatmap.set_xticklabels(tick_vals, fontsize=11)
    ax_heatmap.set_yticks([v + 0.5 for v in tick_vals])
    ax_heatmap.set_yticklabels(tick_vals, fontsize=11)
    ax_heatmap.set_xlabel("Lower SOC (%)", fontsize=13)
    ax_heatmap.set_ylabel("Upper SOC (%)", fontsize=13)
    ax_heatmap.invert_yaxis()
    li, ui = (lower_soc, upper_soc)
    ax_heatmap.scatter(
        li + 0.5,
        ui + 0.5,
        marker="*",
        s=300,
        color="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )
    ax_heatmap.plot([li + 0.5, li + 0.5], [ui + 0.5, 0], "b-", linewidth=1.5, zorder=5)
    ax_heatmap.plot([0, li + 0.5], [ui + 0.5, ui + 0.5], "b-", linewidth=1.5, zorder=5)
    ax_heatmap.text(
        0.31,
        0.8,
        f"({lower_soc}%, {upper_soc}%)",
        transform=ax_heatmap.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
    )
    ax_heatmap.text(
        -0.25,
        1.18,
        "(a)",
        transform=ax_heatmap.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    cmap_cycles = cm.RdBu_r
    cycles = np.arange(1, batt.cycle_life + 1)
    denom = batt.cycle_life - 1 if batt.cycle_life > 1 else 1
    cycle_normalized = np.clip((cycles - 1) / denom, 0, 1)
    colors = [cmap_cycles(v) for v in cycle_normalized]
    all_voltages = []
    for i, cycle in enumerate(range(1, batt.cycle_life + 1)):
        d = batt.cycle_cache.get(cycle, None)
        if d is None or not d.get("valid", False):
            continue
        soc = d["soc"]
        mask = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(mask) == 0:
            continue
        tt = d["t"][mask]
        vv = d["V"][mask]
        ss = soc[mask]
        ax_soc_time.plot(tt, ss, color=colors[i], linewidth=0.8, alpha=0.7)
        ax_soc_voltage.plot(vv, ss, color=colors[i], linewidth=0.8, alpha=0.7)
        ax_voltage_time.plot(tt, vv, color=colors[i], linewidth=0.8, alpha=0.7)
        all_voltages.extend(vv.tolist())
    ax_soc_time.set_xlabel("Time (s)", fontsize=13)
    ax_soc_time.set_ylabel("SOC (%)", fontsize=13)
    ax_soc_time.set_ylim([lower_soc - 2, upper_soc + 2])
    ax_soc_time.grid(True, alpha=0.01)
    ax_soc_time.tick_params(axis="both", labelsize=12)
    ax_soc_time.text(
        -0.21,
        1.15,
        "(b)",
        transform=ax_soc_time.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax_soc_voltage.set_xlabel("Voltage (V)", fontsize=13)
    ax_soc_voltage.set_ylabel("SOC (%)", fontsize=13)
    ax_soc_voltage.set_ylim([lower_soc - 2, upper_soc + 2])
    ax_soc_voltage.grid(True, alpha=0.01)
    ax_soc_voltage.tick_params(axis="both", labelsize=12)
    ax_soc_voltage.text(
        -0.23,
        1.18,
        "(c)",
        transform=ax_soc_voltage.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax_voltage_time.set_xlabel("Time (s)", fontsize=13)
    ax_voltage_time.set_ylabel("Voltage (V)", fontsize=13)
    ax_voltage_time.grid(True, alpha=0.01)
    ax_voltage_time.tick_params(axis="both", labelsize=12)
    if all_voltages:
        v_min, v_max = (float(min(all_voltages)), float(max(all_voltages)))
        padding = (v_max - v_min) * 0.05
        ax_voltage_time.set_ylim(v_min - padding, v_max + padding)
    ax_voltage_time.text(
        -0.23,
        1.18,
        "(d)",
        transform=ax_voltage_time.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    cbar_ax = fig.add_axes([1.0, 0.15, 0.015, 0.7])
    norm = plt.Normalize(vmin=1, vmax=batt.cycle_life)
    sm = cm.ScalarMappable(cmap=cmap_cycles, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cycle Number", fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    return fig


def extract_features_for_battery(
    mat_path: Path, lower_soc: int, upper_soc: int, out_dir: Path
):
    """
    For each valid cycle:
      • Compute 7 SOC features from the charge-phase voltage window.
      • Append the measured discharge capacity.
    Saves CSV directly into out_dir (exactly one .csv per .mat file).
    """
    batt = XJTUBattery(str(mat_path))
    rows = []
    for cyc in range(1, batt.cycle_life + 1):
        d = batt.cycle_cache.get(cyc, None)
        if d is None or not d.get("valid", False):
            continue
        cap_ah = d.get("y_cap", np.nan)
        cap_ah = float(cap_ah) if np.isfinite(cap_ah) else 0.0
        a, b = _window_indices_from_soc(d["soc"], lower_soc, upper_soc)
        if a is None:
            continue
        soc_feats = _calc_features_fast_from_ab(d, a, b)
        if soc_feats is None:
            continue
        row = {k: float(soc_feats[k]) for k in SOC_FEATURE_NAMES}
        row[LAST_COL_NAME] = cap_ah
        rows.append(row)
    df = pd.DataFrame(rows, columns=FEATURE_NAMES_OUT + [LAST_COL_NAME])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{mat_path.stem}.csv"
    df.to_csv(out_file, index=False)
    return (len(df), out_file)


def discover_mat_files():
    files = []
    for bdir in BATCH_DIRS:
        if not bdir.exists():
            print(f"[WARN] Missing batch folder: {bdir}")
            continue
        files.extend(sorted(bdir.glob("*.mat")))
    return files


def find_reference_path():
    for p in BATCH_DIRS:
        if p.name.lower() == "batch-1":
            candidate = p / REFERENCE_FILE
            if candidate.exists():
                return candidate
            break
    return None


def option_1_psbo():
    print("\n" + "=" * 100)
    print("[1] PS-BO WINDOW SELECTION + FEATURE EXTRACTION (native)")
    print("=" * 100)
    print(f"Features: {FEATURE_NAMES_OUT} + Capacity")
    print(f"Nominal capacity : {NOMINAL_CAPACITY_AH} Ah")
    ref_path = find_reference_path()
    if ref_path is None:
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE} inside Batch-1 folder."
        )
    all_mat_files = discover_mat_files()
    print(f"\nFound total MAT files : {len(all_mat_files)}")
    print("\n[1] PS-BO window selection using reference cell:")
    print(f"    Reference: {ref_path}")
    ref_batt = XJTUBattery(str(ref_path))
    best, score_map, axes, analysis = psbo_select_window_fast(
        ref_batt, NOMINAL_CAPACITY_AH, show_progress=True
    )
    l_opt, u_opt = (int(best["lower_soc"]), int(best["upper_soc"]))
    print(f"\n    -> Best window: {l_opt}% - {u_opt}% | score={best['score']:.6f}")
    print(f"    -> Search time: {analysis['search_time_sec']:.2f} sec")
    print("\n[2] Saving 2x2 figure (PNG+PDF+SVG) + Figure(a-d) CSVs for PS-BO...")
    FIG_DIR = PSBO_ROOT / "Figures_2x2"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_CSV_DIR = PSBO_ROOT / "Figure_CSVs_(a-d)"
    FIG_CSV_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_comprehensive_visualization(best, score_map, axes, ref_batt)
    png_path = FIG_DIR / f"{ref_batt.battery_name}_psbo_comprehensive.png"
    pdf_path = FIG_DIR / f"{ref_batt.battery_name}_psbo_comprehensive.pdf"
    svg_path = FIG_DIR / f"{ref_batt.battery_name}_psbo_comprehensive.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved PNG: {png_path}")
    print(f"    ✓ Saved PDF: {pdf_path}")
    print(f"    ✓ Saved SVG: {svg_path}")
    save_figure_csvs(ref_batt, best, score_map, FIG_CSV_DIR)
    print(f"    ✓ Saved figure CSVs in: {FIG_CSV_DIR}")
    print(
        f"\n[3] Extracting 7 features + Capacity — PS-BO window ({l_opt}%-{u_opt}%)..."
    )
    print(f"    → PS-BO Window Selected → {PSBO_ROOT}")
    PSBO_ROOT.mkdir(parents=True, exist_ok=True)
    win_ok = 0
    for mat_path in all_mat_files:
        try:
            nrows, out_csv = extract_features_for_battery(
                mat_path, l_opt, u_opt, PSBO_ROOT
            )
            print(f"        ✓ {mat_path.name} → {nrows} cycles → {out_csv.name}")
            win_ok += 1
        except Exception as e:
            print(f"        [ERROR] {mat_path.name}: {e}")
    print(f"    Saved {win_ok} CSV files for PS-BO Window Selected")
    print("\nDONE — PS-BO window extraction complete")
    return {"best": best, "lower": l_opt, "upper": u_opt}


def option_2_fixed_windows():
    print("\n" + "=" * 100)
    print("[2] FIXED SOC WINDOWS (native)")
    print("=" * 100)
    all_mat_files = discover_mat_files()
    for lower, upper, out_root in FIXED_WINDOWS:
        out_root.mkdir(parents=True, exist_ok=True)
        print(
            f"\n    → {lower} %- {upper}% Window Selected ({lower}%–{upper}%) → {out_root}"
        )
        win_ok = 0
        for mat_path in all_mat_files:
            try:
                nrows, out_csv = extract_features_for_battery(
                    mat_path, lower, upper, out_root
                )
                print(f"        ✓ {mat_path.name} → {nrows} cycles → {out_csv.name}")
                win_ok += 1
            except Exception as e:
                print(f"        [ERROR] {mat_path.name}: {e}")
        print(f"    Saved {win_ok} CSV files for {lower} %- {upper}% Window Selected")
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
