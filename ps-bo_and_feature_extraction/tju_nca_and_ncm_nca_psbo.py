"""tju_nca_psbo: native dataset processing and retained experiments."""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
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
LAST_COL_NAME = "Capacity"
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
FIXED_WINDOWS = [
    (0, 100, "0 %- 100% Window Selected"),
    (53, 97, "NCM 53 %- 97% Window Selected"),
    (13, 83, "LCO 13 %- 87% Window Selected"),
    (52, 91, "LFP 52 %- 91% Window Selected"),
    (21, 51, "NA_ion 21 %- 51% Window Selected"),
    (20, 80, "20 %- 80% Window Selected"),
    (25, 75, "25 %- 75% Window Selected"),
    (30, 70, "30 %- 70% Window Selected"),
    (40, 60, "40 %- 60% Window Selected"),
]
CONFIGS = [
    {
        "name": "NCA",
        "data_dir": Path("D:\\Feature Extarction\\TJU\\NCA\\Data"),
        "nominal_capacity": 3.5,
        "reference_cell": "CY45-05_1-#1.csv",
        "apply_batch1_threshold": True,
        "cell_list": [
            "CY25-05_1-#1.csv",
            "CY25-05_1-#2.csv",
            "CY25-05_1-#3.csv",
            "CY25-05_1-#4.csv",
            "CY25-05_1-#5.csv",
            "CY25-05_1-#6.csv",
            "CY25-05_1-#7.csv",
            "CY25-05_1-#10.csv",
            "CY25-05_1-#11.csv",
            "CY25-05_1-#12.csv",
            "CY25-05_1-#13.csv",
            "CY25-05_1-#14.csv",
            "CY25-05_1-#15.csv",
            "CY35-05_1-#1.csv",
            "CY35-05_1-#2.csv",
            "CY45-05_1-#2.csv",
            "CY45-05_1-#3.csv",
            "CY45-05_1-#4.csv",
            "CY45-05_1-#5.csv",
            "CY45-05_1-#6.csv",
            "CY45-05_1-#7.csv",
            "CY45-05_1-#8.csv",
            "CY45-05_1-#9.csv",
            "CY45-05_1-#10.csv",
            "CY45-05_1-#11.csv",
            "CY45-05_1-#12.csv",
            "CY45-05_1-#16.csv",
            "CY45-05_1-#18.csv",
            "CY45-05_1-#19.csv",
            "CY45-05_1-#20.csv",
        ],
    },
    {
        "name": "NCM_NCA",
        "data_dir": Path("D:\\Feature Extarction\\TJU\\NCM_NCA\\Data"),
        "nominal_capacity": 2.5,
        "reference_cell": "CY25-05_2-#3.csv",
        "apply_batch1_threshold": False,
        "cell_list": [
            "CY25-05_1-#1.csv",
            "CY25-05_1-#2.csv",
            "CY25-05_1-#3.csv",
            "CY25-05_2-#1.csv",
            "CY25-05_2-#2.csv",
            "CY25-05_4-#1.csv",
            "CY25-05_4-#2.csv",
            "CY25-05_4-#3.csv",
        ],
    },
]


def get_memory_mb():
    """Return current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


class TJUBattery:

    def __init__(self, csv_path: str, apply_batch1_threshold: bool):
        self.csv_path = str(csv_path)
        self.battery_name = os.path.basename(self.csv_path).replace(".csv", "")
        self.data_df = pd.read_csv(self.csv_path)
        self.apply_batch1_threshold = bool(apply_batch1_threshold)
        if "cycle number" not in self.data_df.columns:
            raise ValueError(f"Missing column 'cycle number' in {self.csv_path}")
        self.cycle_index = np.unique(self.data_df["cycle number"].values)
        self.cycle_life = len(self.cycle_index)
        self.cycle_cache: dict = {}
        self._build_cycle_cache()

    def _build_cycle_cache(self):
        grouped = self.data_df.groupby("cycle number", sort=False)
        for cyc in self.cycle_index:
            try:
                cycle_data = grouped.get_group(cyc)
            except Exception:
                self.cycle_cache[cyc] = None
                continue
            cap_ah = np.nan
            if "Q discharge/mA.h" in cycle_data.columns:
                discharge_data = cycle_data[cycle_data["Q discharge/mA.h"] > 0]
                if not discharge_data.empty:
                    cap_ah = (
                        float(np.max(discharge_data["Q discharge/mA.h"].values))
                        / 1000.0
                    )
            charge_data = cycle_data[cycle_data["<I>/mA"] > 0].copy()
            if charge_data.empty or len(charge_data) < 10:
                self.cycle_cache[cyc] = {"valid": False, "cap_ah": cap_ah}
                continue
            required_cols = {"time/s", "Q charge/mA.h", "Ecell/V"}
            if not required_cols.issubset(set(charge_data.columns)):
                self.cycle_cache[cyc] = {"valid": False, "cap_ah": cap_ah}
                continue
            charge_data = charge_data.sort_values("time/s").reset_index(drop=True)
            capacity = charge_data["Q charge/mA.h"].values / 1000.0
            voltage = charge_data["Ecell/V"].values
            time_arr = charge_data["time/s"].values
            valid_mask = (
                np.isfinite(capacity) & np.isfinite(voltage) & np.isfinite(time_arr)
            )
            capacity = capacity[valid_mask]
            voltage = voltage[valid_mask]
            time_arr = time_arr[valid_mask]
            if len(capacity) < 10:
                self.cycle_cache[cyc] = {"valid": False, "cap_ah": cap_ah}
                continue
            time_arr = time_arr - time_arr[0]
            capacity_normalized = capacity - capacity[0]
            max_capacity = (
                capacity_normalized[-1] if capacity_normalized[-1] > 0 else 1.0
            )
            soc = capacity_normalized / max_capacity * 100.0
            V = voltage.astype(float)
            t = time_arr.astype(float)
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
            ct[1:] = np.cumsum(t)
            ct2 = np.zeros(N + 1)
            ct2[1:] = np.cumsum(t**2)
            ctv = np.zeros(N + 1)
            ctv[1:] = np.cumsum(t * V)
            self.cycle_cache[cyc] = {
                "valid": True,
                "cap_ah": cap_ah,
                "V": V,
                "t": t,
                "soc": soc,
                "N": N,
                "capacity_ah": capacity,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "ct": ct,
                "ct2": ct2,
                "ctv": ctv,
            }

    def get_cycle_data(self, cycle: int):
        d = self.cycle_cache.get(cycle, None)
        if d is None or not d.get("valid", False):
            return None
        time_min = d["t"] / 60.0
        return {"capacity": d["capacity_ah"], "voltage": d["V"], "time": time_min}


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
    pk = hist / N
    pk = pk[pk > 0]
    return float(-np.sum(pk * np.log2(pk))) if len(pk) else 0.0


def _calc_soc_features_fast(d: dict, a: int, b: int):
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
    battery: TJUBattery, lower: int, upper: int, nominal_capacity: float
):
    cycles = battery.cycle_index
    y = np.full(len(cycles), np.nan, dtype=float)
    x_by_f = {f: np.full(len(cycles), np.nan, dtype=float) for f in FEATURE_NAMES}
    for i, cyc in enumerate(cycles):
        d = battery.cycle_cache.get(cyc, None)
        if d is None:
            continue
        cap_ah = d.get("cap_ah", np.nan)
        if battery.apply_batch1_threshold and np.isfinite(cap_ah) and (cap_ah < 1.5):
            continue
        y[i] = cap_ah / nominal_capacity if np.isfinite(cap_ah) else np.nan
        if not d.get("valid", False):
            continue
        a, b = _window_indices_from_soc(d["soc"], lower, upper)
        if a is None or b - a < MIN_POINTS_GRID:
            continue
        feats = _calc_soc_features_fast(d, a, b)
        if feats is None:
            continue
        for f in FEATURE_NAMES:
            x_by_f[f][i] = feats.get(f, np.nan)
    r_by_feature = {}
    p_by_feature = {}
    n_by_feature = {}
    for f in FEATURE_NAMES:
        x = x_by_f[f]
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
    battery: TJUBattery, nominal_capacity: float, show_progress: bool = True
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
        desc=f"PS-BO search [{battery.battery_name}]",
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
            battery, lower_soc, upper_soc, nominal_capacity
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
                res = evaluate_window_score_fast(battery, l, u, nominal_capacity)
                if res is None:
                    continue
                score_map[u, l] = res["score"]
                if res["score"] > best["score"]:
                    best.update(res)
    if best["lower_soc"] is None:
        raise RuntimeError(f"No valid SOC window found for {battery.battery_name}")
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
    batt: TJUBattery, best: dict, score_map: np.ndarray, output_dir: Path
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
        d = batt.get_cycle_data(cycle)
        if d is None:
            continue
        cap = d["capacity"]
        if len(cap) == 0:
            continue
        cap_norm = cap - cap[0]
        max_cap = cap_norm[-1] if cap_norm[-1] > 0 else 1.0
        soc = cap_norm / max_cap * 100.0
        mask = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(mask) == 0:
            continue
        t_sec = d["time"][mask] * 60.0
        v = d["voltage"][mask]
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
    best: dict, score_map: np.ndarray, axes: tuple, batt: TJUBattery, title_text: str
):
    lower_vals, upper_vals = axes
    lower_soc = int(best["lower_soc"])
    upper_soc = int(best["upper_soc"])
    fig = plt.figure(figsize=(6, 5.5))
    fig.suptitle(title_text, fontsize=12, y=1.05)
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
        0.35,
        0.78,
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
    cycle_normalized = np.clip(
        (cycles - 0) / (batt.cycle_life - 0 if batt.cycle_life != 0 else 1), 0, 1
    )
    colors = [cmap_cycles(v) for v in cycle_normalized]
    all_voltages = []
    for i, cycle in enumerate(range(1, batt.cycle_life + 1)):
        d = batt.get_cycle_data(cycle)
        if d is None:
            continue
        cap = d["capacity"]
        if len(cap) == 0:
            continue
        cap_norm = cap - cap[0]
        max_cap = cap_norm[-1] if cap_norm[-1] > 0 else 1.0
        soc = cap_norm / max_cap * 100.0
        mask = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(mask) == 0:
            continue
        ax_soc_time.plot(
            d["time"][mask] * 60.0, soc[mask], color=colors[i], linewidth=0.8, alpha=0.7
        )
        ax_soc_voltage.plot(
            d["voltage"][mask], soc[mask], color=colors[i], linewidth=0.8, alpha=0.7
        )
        tt = d["time"][mask] * 60.0
        vv = d["voltage"][mask]
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
    cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
    norm = plt.Normalize(vmin=0, vmax=batt.cycle_life)
    sm = cm.ScalarMappable(cmap=cmap_cycles, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cycle Number", fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    return fig


def extract_features_for_cell(
    csv_path: Path,
    lower_soc: int,
    upper_soc: int,
    out_dir: Path,
    apply_batch1_threshold: bool,
):
    battery = TJUBattery(str(csv_path), apply_batch1_threshold=apply_batch1_threshold)
    rows = []
    for cyc in battery.cycle_index:
        d = battery.cycle_cache.get(cyc, None)
        if d is None:
            continue
        cap = d.get("cap_ah", np.nan)
        cap = float(cap) if np.isfinite(cap) else 0.0
        if apply_batch1_threshold and cap < 1.5:
            continue
        if not d.get("valid", False):
            continue
        a, b = _window_indices_from_soc(d["soc"], lower_soc, upper_soc)
        if a is None:
            continue
        soc_feats = _calc_soc_features_fast(d, a, b)
        if soc_feats is None:
            continue
        row = {k: float(soc_feats[k]) for k in SOC_FEATURE_NAMES}
        row[LAST_COL_NAME] = cap
        rows.append(row)
    df = pd.DataFrame(rows, columns=FEATURE_NAMES_OUT + [LAST_COL_NAME])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / csv_path.name
    df.to_csv(out_file, index=False)
    return (len(df), out_file)


def _run_group_psbo(cfg):
    name = cfg["name"]
    data_dir = Path(cfg["data_dir"])
    nominal = float(cfg["nominal_capacity"])
    ref_cell = cfg["reference_cell"]
    apply_thr = bool(cfg["apply_batch1_threshold"])
    cell_list = list(cfg["cell_list"])
    print("\n" + "-" * 100)
    print(f"TJU RUN: {name} — PS-BO WINDOW (native pipeline)")
    print("-" * 100)
    print(f"Input folder : {data_dir}")
    print(f"Nominal cap  : {nominal} Ah")
    print(f"Reference    : {ref_cell}")
    print(f"Threshold    : {('ON' if apply_thr else 'OFF')}")
    print(f"Features     : {FEATURE_NAMES_OUT} + Capacity")
    ref_path = data_dir / ref_cell
    if not ref_path.exists():
        print(f"[ERROR] Missing reference file: {ref_path}")
        return None
    print("\n[1] PS-BO window selection on reference cell...")
    ref_batt = TJUBattery(str(ref_path), apply_batch1_threshold=apply_thr)
    best, score_map, axes, analysis = psbo_select_window_fast(
        ref_batt, nominal, show_progress=True
    )
    l_opt, u_opt = (int(best["lower_soc"]), int(best["upper_soc"]))
    print(f"    -> Best window: {l_opt}% - {u_opt}% | score={best['score']:.6f}")
    psbo_root = data_dir / "7PS-BO Window Selected"
    psbo_root.mkdir(parents=True, exist_ok=True)
    print("\n[2] Saving 2x2 figure + Figure CSVs for PS-BO window...")
    if name == "NCA":
        title_text = "NCA (CY45-05_1-#1)"
    elif name == "NCM_NCA":
        title_text = "NCM_NCA (CY25-05_2-#3)"
    else:
        title_text = f"{name} ({ref_batt.battery_name})"
    fig = plot_comprehensive_visualization(best, score_map, axes, ref_batt, title_text)
    fig_png = psbo_root / f"{Path(ref_cell).stem}_PSBO_2x2.png"
    fig_pdf = psbo_root / f"{Path(ref_cell).stem}_PSBO_2x2.pdf"
    fig_svg = psbo_root / f"{Path(ref_cell).stem}_PSBO_2x2.svg"
    fig.savefig(fig_png, dpi=600, bbox_inches="tight")
    fig.savefig(fig_pdf, dpi=600, bbox_inches="tight")
    fig.savefig(fig_svg, bbox_inches="tight")
    plt.close(fig)
    save_figure_csvs(ref_batt, best, score_map, psbo_root)
    print(f"    Saved: {fig_png}")
    print(f"    Saved: {fig_pdf}")
    print(f"    Saved: {fig_svg}")
    print(f"    Saved Figure CSVs in: {psbo_root}")
    print("\n[3] Extracting 7 features + Capacity for the PS-BO window...")
    print(f"\n    → 7PS-BO Window Selected ({l_opt}%-{u_opt}%) → {psbo_root}")
    win_ok = 0
    for cell_name in cell_list:
        cell_p = data_dir / cell_name
        if not cell_p.exists():
            continue
        try:
            nrows, out_csv = extract_features_for_cell(
                cell_p, l_opt, u_opt, psbo_root, apply_thr
            )
            print(f"        ✓ {cell_name} → {nrows} cycles")
            win_ok += 1
        except Exception as e:
            print(f"        [ERROR] {cell_name}: {e}")
    print(f"    Saved {win_ok} CSV files for 7PS-BO Window Selected")
    print(f"\nDONE [{name}] — PS-BO window extraction complete")
    return {"name": name, "best": best, "lower": l_opt, "upper": u_opt}


def option_1_psbo():
    print("\n" + "=" * 100)
    print("[1] PS-BO WINDOW SELECTION + FEATURE EXTRACTION (native, both groups)")
    print("=" * 100)
    results = {}
    for cfg in CONFIGS:
        r = _run_group_psbo(cfg)
        if r is not None:
            results[cfg["name"]] = r
    return results


def _run_group_fixed(cfg):
    name = cfg["name"]
    data_dir = Path(cfg["data_dir"])
    apply_thr = bool(cfg["apply_batch1_threshold"])
    cell_list = list(cfg["cell_list"])
    print("\n" + "-" * 100)
    print(f"TJU RUN: {name} — FIXED SOC WINDOWS (native pipeline)")
    print("-" * 100)
    for low, up, fname in FIXED_WINDOWS:
        fldr = data_dir / fname
        fldr.mkdir(parents=True, exist_ok=True)
        print(f"\n    → {fname} ({low}%–{up}%) → {fldr}")
        win_ok = 0
        for cell_name in cell_list:
            cell_p = data_dir / cell_name
            if not cell_p.exists():
                continue
            try:
                nrows, out_csv = extract_features_for_cell(
                    cell_p, low, up, fldr, apply_thr
                )
                print(f"        ✓ {cell_name} → {nrows} cycles")
                win_ok += 1
            except Exception as e:
                print(f"        [ERROR] {cell_name}: {e}")
        print(f"    Saved {win_ok} CSV files for {fname}")
    print(f"\nDONE [{name}] — fixed windows complete")


def option_2_fixed_windows():
    print("\n" + "=" * 100)
    print("[2] FIXED SOC WINDOWS (native, both groups)")
    print("=" * 100)
    for cfg in CONFIGS:
        _run_group_fixed(cfg)


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
