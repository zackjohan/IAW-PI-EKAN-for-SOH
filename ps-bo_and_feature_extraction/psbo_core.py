"""Shared PS-BO scoring, feature extraction, plotting, and retained perturbations."""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib import rcParams
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import psutil

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
FEATURE_NAMES = SOC_FEATURE_NAMES
DEFAULT_LAST_COL_NAME = "Normalized Capacity"
SOC_MIN = 0
SOC_MAX = 100
PSBO_N_CALLS = 120
PSBO_N_RANDOM_STARTS = 25
PSBO_RANDOM_STATE = 42
MIN_POINTS_GRID = 2
MIN_CYCLES_GRID = 2
MIN_FEATURES_OK = 6
USE_ABS_CORR = True
MIN_POINTS_FEATURES = 10
MIN_WINDOW_COVERAGE = 0.99
DEFAULT_CAP_THRESHOLD = 1.5


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def longest_contiguous_segment(indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return indices
    best = [indices[0]]
    cur = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            cur.append(indices[i])
        else:
            if len(cur) > len(best):
                best = cur
            cur = [indices[i]]
    if len(cur) > len(best):
        best = cur
    return np.array(best, dtype=int)


def downsample_to_hz(t: np.ndarray, target_hz: float = 1.0) -> np.ndarray:
    """
    MODULE 6 — frequency-parameterised decimation.

    The minimum inter-sample gap is 0.95 / target_hz seconds, so
    target_hz = 1.0 gives a 0.95 s gap and reproduces downsample_to_1hz
    EXACTLY. Nothing changes unless a different rate is requested.
    """
    if len(t) == 0:
        return np.array([], dtype=int)
    gap = 0.95 / float(target_hz)
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= gap:
            keep.append(i)
    return np.array(keep, dtype=int)


def apply_bias_drift_to_current(curr: np.ndarray, bias_crate: float) -> np.ndarray:
    """Scenario A — constant current-sensor offset applied BEFORE integration."""
    return curr + bias_crate


def apply_init_error_to_soc(soc: np.ndarray, init_error_percent: float) -> np.ndarray:
    """Scenario B — rigid SOC_0 offset applied AFTER integration."""
    return np.clip(soc + init_error_percent, 0.0, 100.0).astype(float)


def apply_random_walk_to_soc(
    soc: np.ndarray, sigma_step: float, rng: np.random.Generator
) -> np.ndarray:
    """Scenario C — Brownian drift; monotonicity intentionally not enforced."""
    steps = rng.normal(0.0, sigma_step, size=len(soc))
    drift = np.cumsum(steps)
    return np.clip(soc + drift, 0.0, 100.0).astype(float)


def apply_spike_noise_to_soc(
    soc: np.ndarray, spike_prob: float, spike_mag: float, rng: np.random.Generator
) -> np.ndarray:
    """Scenario D — impulsive glitches; monotonicity intentionally not enforced."""
    soc_noisy = soc.copy().astype(float)
    spike_mask = rng.random(size=len(soc_noisy)) < spike_prob
    if np.any(spike_mask):
        spikes = rng.uniform(-spike_mag, spike_mag, size=int(np.sum(spike_mask)))
        soc_noisy[spike_mask] += spikes
    return np.clip(soc_noisy, 0.0, 100.0).astype(float)


def slice_soc_window(cached_cycle: dict, lower_soc, upper_soc, min_points=2):
    soc, V, t = (cached_cycle["soc"], cached_cycle["V"], cached_cycle["t"])
    mask = (soc >= lower_soc) & (soc <= upper_soc)
    if np.sum(mask) < min_points:
        return None
    return {"soc": soc[mask], "V": V[mask], "t": t[mask]}


def calc_7_features(window: dict):
    if window is None:
        return None
    t, V = (window["t"], window["V"])
    n = len(V)
    if n < MIN_POINTS_FEATURES:
        return None
    feats = {}
    feats["ChargingTime"] = float(t[-1] - t[0])
    sum_t = np.sum(t)
    sum_V = np.sum(V)
    sum_tV = np.sum(t * V)
    sum_t2 = np.sum(t**2)
    den = n * sum_t2 - sum_t**2
    feats["VoltageSlope"] = (
        float((n * sum_tV - sum_t * sum_V) / den) if abs(den) > 1e-12 else 0.0
    )
    feats["VoltageStd"] = float(np.std(V, ddof=1)) if n > 1 else 0.0
    try:
        k = float(stats.kurtosis(V, fisher=True, bias=False))
        feats["VoltageKurtosis"] = k if np.isfinite(k) else 0.0
    except Exception:
        feats["VoltageKurtosis"] = 0.0
    try:
        s = float(stats.skew(V, bias=False))
        feats["VoltageSkewness"] = s if np.isfinite(s) else 0.0
    except Exception:
        feats["VoltageSkewness"] = 0.0
    try:
        B = min(100, max(1, n // 10))
        hist, _ = np.histogram(V, bins=B)
        pk = hist / n
        pk = pk[pk > 0]
        feats["VoltageEntropy"] = float(-np.sum(pk * np.log2(pk))) if len(pk) else 0.0
    except Exception:
        feats["VoltageEntropy"] = 0.0
    try:
        X = np.column_stack([t**2, t, np.ones(n)])
        coeffs = np.linalg.lstsq(X, V, rcond=None)[0]
        feats["VoltageCurvature"] = float(coeffs[0])
    except Exception:
        feats["VoltageCurvature"] = 0.0
    return feats


def evaluate_window(lower: int, upper: int, cell, cap_map: dict):
    n_total = cell.n_cycles
    if n_total < 10:
        return None
    y = np.full(n_total, np.nan, dtype=float)
    for i, cyc_id in enumerate(cell.cycle_ids):
        if cyc_id in cap_map:
            y[i] = cap_map[cyc_id]
    x_by_feat = {name: np.full(n_total, np.nan, dtype=float) for name in FEATURE_NAMES}
    valid_cycles = 0
    for k in range(n_total):
        if not np.isfinite(y[k]):
            continue
        w = slice_soc_window(
            cell.charge_segments[k], lower, upper, min_points=MIN_POINTS_GRID
        )
        if w is None:
            continue
        feats = calc_7_features(w)
        if feats is None:
            continue
        valid_cycles += 1
        for name in FEATURE_NAMES:
            x_by_feat[name][k] = feats.get(name, np.nan)
    if n_total == 0:
        return None
    n_labeled = int(np.sum(np.isfinite(y)))
    if n_labeled < MIN_CYCLES_GRID:
        return None
    coverage = valid_cycles / n_labeled
    if coverage < MIN_WINDOW_COVERAGE:
        return None
    r_by_feature = {}
    for name in FEATURE_NAMES:
        x = x_by_feat[name]
        mask = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(mask)) < MIN_CYCLES_GRID:
            continue
        try:
            r, _ = stats.pearsonr(x[mask], y[mask])
            if np.isfinite(r):
                r_by_feature[name] = float(r)
        except Exception:
            continue
    if len(r_by_feature) < MIN_FEATURES_OK:
        return None
    rs = np.array(list(r_by_feature.values()), dtype=float)
    score = float(np.mean(np.abs(rs))) if USE_ABS_CORR else float(np.mean(rs))
    return {
        "lower_soc": int(lower),
        "upper_soc": int(upper),
        "score": score,
        "coverage": float(coverage),
    }


def psbo_optimize(
    cell, cap_map: dict, show_progress: bool = True, desc_suffix: str = ""
):
    score_map = np.full((101, 101), np.nan, dtype=float)
    best = {"lower_soc": None, "upper_soc": None, "score": -np.inf, "coverage": 0.0}
    space = [
        Integer(SOC_MIN, SOC_MAX, name="lower_soc"),
        Integer(SOC_MIN, SOC_MAX, name="upper_soc"),
    ]
    pbar = tqdm(
        total=PSBO_N_CALLS,
        desc=f"PS-BO [{cell.name}]{desc_suffix}",
        disable=not show_progress,
    )

    @use_named_args(space)
    def objective(lower_soc, upper_soc):
        pbar.update(1)
        if upper_soc < lower_soc:
            return 1.0
        res = evaluate_window(lower_soc, upper_soc, cell, cap_map)
        if res is None:
            score_map[int(upper_soc), int(lower_soc)] = np.nan
            return 1.0
        score_map[int(upper_soc), int(lower_soc)] = res["score"]
        if res["score"] > best["score"]:
            best.update(res)
        return -res["score"]

    t0 = time.perf_counter()
    gp_minimize(
        objective,
        space,
        n_calls=PSBO_N_CALLS,
        n_initial_points=PSBO_N_RANDOM_STARTS,
        random_state=PSBO_RANDOM_STATE,
        acq_func="EI",
    )
    pbar.close()
    t1 = time.perf_counter()
    if best["lower_soc"] is None:
        for l in range(0, 101, 5):
            for u in range(l, 101, 5):
                res = evaluate_window(l, u, cell, cap_map)
                if res is None:
                    continue
                score_map[u, l] = res["score"]
                if res["score"] > best["score"]:
                    best.update(res)
    if best["lower_soc"] is None:
        raise RuntimeError(
            "No valid SOC window found. (Try lowering MIN_WINDOW_COVERAGE / thresholds.)"
        )
    analysis = {"search_time_sec": float(t1 - t0)}
    return (best, score_map, (np.arange(0, 101), np.arange(0, 101)), analysis)


def plot_2x2(
    best: dict, score_map: np.ndarray, axes: tuple, cell, title_text: str = ""
):
    lower_vals, upper_vals = axes
    lower_soc = best["lower_soc"]
    upper_soc = best["upper_soc"]
    fig = plt.figure(figsize=(6, 5.5))
    fig.suptitle(title_text, fontsize=12, y=1.05)
    gs = fig.add_gridspec(
        2, 2, hspace=0.45, wspace=0.4, left=0.08, right=0.95, top=0.95, bottom=0.08
    )
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_soc_t = fig.add_subplot(gs[0, 1])
    ax_soc_v = fig.add_subplot(gs[1, 0])
    ax_v_t = fig.add_subplot(gs[1, 1])
    score_map_full = np.full((101, 101), np.nan, dtype=float)
    for u in upper_vals:
        for l in lower_vals:
            score_map_full[int(u), int(l)] = score_map[int(u), int(l)]
    mask = np.zeros_like(score_map_full, dtype=bool)
    for u in range(0, 101):
        for l in range(0, 101):
            if u < l:
                mask[u, l] = True
    sns.heatmap(
        score_map_full,
        mask=mask,
        cmap="RdBu_r",
        cbar_kws={"label": "Aggregate Score", "shrink": 0.8},
        ax=ax_hm,
        xticklabels=False,
        yticklabels=False,
    )
    tick_vals = list(range(0, 101, 20))
    pos = [v + 0.5 for v in tick_vals]
    ax_hm.set_xticks(pos)
    ax_hm.set_xticklabels(tick_vals, fontsize=11)
    ax_hm.set_yticks(pos)
    ax_hm.set_yticklabels(tick_vals, fontsize=11)
    ax_hm.set_xlabel("Lower SOC (%)", fontsize=13)
    ax_hm.set_ylabel("Upper SOC (%)", fontsize=13)
    ax_hm.invert_yaxis()
    li, ui = (int(lower_soc), int(upper_soc))
    ax_hm.scatter(
        li + 0.5,
        ui + 0.5,
        marker="*",
        s=300,
        color="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
    )
    ax_hm.text(
        0.35,
        0.78,
        f"({lower_soc}%, {upper_soc}%)",
        transform=ax_hm.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
    )
    ax_hm.text(
        -0.25,
        1.18,
        "(a)",
        transform=ax_hm.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    cmap_cycles = cm.RdBu_r
    n_cycles = cell.n_cycles if cell.n_cycles > 0 else 1
    cycnums = np.arange(1, n_cycles + 1)
    normed = np.clip((cycnums - 0) / (n_cycles if n_cycles != 0 else 1), 0, 1)
    colors = [cmap_cycles(v) for v in normed]
    for i in range(n_cycles):
        arr = cell.charge_segments[i]
        soc, t, V = (arr["soc"], arr["t"], arr["V"])
        m = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(m) > 0:
            ax_soc_t.plot(t[m], soc[m], color=colors[i], linewidth=0.8, alpha=0.7)
    ax_soc_t.set_xlabel("Time (s)", fontsize=13)
    ax_soc_t.set_ylabel("SOC (%)", fontsize=13)
    ax_soc_t.set_ylim([lower_soc - 2, upper_soc + 2])
    ax_soc_t.grid(True, alpha=0.01)
    ax_soc_t.tick_params(axis="both", labelsize=12)
    ax_soc_t.text(
        -0.21,
        1.15,
        "(b)",
        transform=ax_soc_t.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    for i in range(n_cycles):
        arr = cell.charge_segments[i]
        soc, V = (arr["soc"], arr["V"])
        m = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(m) > 0:
            ax_soc_v.plot(V[m], soc[m], color=colors[i], linewidth=0.8, alpha=0.7)
    ax_soc_v.set_xlabel("Voltage (V)", fontsize=13)
    ax_soc_v.set_ylabel("SOC (%)", fontsize=13)
    ax_soc_v.set_ylim([lower_soc - 2, upper_soc + 2])
    ax_soc_v.grid(True, alpha=0.01)
    ax_soc_v.tick_params(axis="both", labelsize=12)
    ax_soc_v.text(
        -0.23,
        1.18,
        "(c)",
        transform=ax_soc_v.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    allV = []
    for i in range(n_cycles):
        arr = cell.charge_segments[i]
        soc, t, V = (arr["soc"], arr["t"], arr["V"])
        m = (soc >= lower_soc) & (soc <= upper_soc)
        if np.sum(m) > 0:
            ax_v_t.plot(t[m], V[m], color=colors[i], linewidth=0.8, alpha=0.7)
            allV.extend(V[m].tolist())
    ax_v_t.set_xlabel("Time (s)", fontsize=13)
    ax_v_t.set_ylabel("Voltage (V)", fontsize=13)
    ax_v_t.grid(True, alpha=0.01)
    ax_v_t.tick_params(axis="both", labelsize=12)
    if allV:
        vmin, vmax = (float(np.min(allV)), float(np.max(allV)))
        pad = (vmax - vmin) * 0.05
        ax_v_t.set_ylim(vmin - pad, vmax + pad)
    ax_v_t.text(
        -0.23,
        1.18,
        "(d)",
        transform=ax_v_t.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=0, vmax=n_cycles)
    sm = cm.ScalarMappable(cmap=cmap_cycles, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Cycle Number", fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    return fig


def save_figure_3formats(fig, base_path_no_ext):
    fig.savefig(base_path_no_ext + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base_path_no_ext + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(base_path_no_ext + ".svg", bbox_inches="tight")
    plt.close(fig)


def extract_features_csv(
    cell,
    cap_map: dict,
    lower_soc: int,
    upper_soc: int,
    diag_interp=None,
    cap_threshold: float = DEFAULT_CAP_THRESHOLD,
    last_col_name: str = DEFAULT_LAST_COL_NAME,
) -> pd.DataFrame:
    rows = []
    for i in range(cell.n_cycles):
        cyc_id = cell.cycle_ids[i]
        if cyc_id not in cap_map:
            continue
        cap_raw = cap_map[cyc_id]
        if not np.isfinite(cap_raw) or cap_raw > cap_threshold:
            continue
        w = slice_soc_window(
            cell.charge_segments[i],
            lower_soc,
            upper_soc,
            min_points=MIN_POINTS_FEATURES,
        )
        if w is None:
            continue
        soc_feats = calc_7_features(w)
        if soc_feats is None:
            continue
        if diag_interp is not None:
            cap_out = diag_interp(float(cyc_id))
        else:
            cap_out = cap_raw
        row = {name: float(soc_feats.get(name, 0.0)) for name in SOC_FEATURE_NAMES}
        row[last_col_name] = float(cap_out)
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_NAMES_OUT + [last_col_name])


def write_window_report(
    path: str,
    header_lines: list,
    best: dict,
    analysis: dict,
    cell,
    extra_lines: list = None,
):
    with open(path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line.rstrip("\n") + "\n")
        f.write(f"REFERENCE: {cell.name}\n")
        f.write(f"AGING CYCLES EXTRACTED: {cell.n_cycles}\n")
        f.write(f"BEST LOWER SOC: {int(best['lower_soc'])}\n")
        f.write(f"BEST UPPER SOC: {int(best['upper_soc'])}\n")
        f.write(f"BEST SCORE (mean |r|): {best['score']:.8f}\n")
        f.write(f"COVERAGE: {best.get('coverage', 0.0) * 100:.4f}%\n")
        f.write(f"FEATURES: {FEATURE_NAMES_OUT}  (7 SOC features only)\n")
        for line in extra_lines or []:
            f.write(line.rstrip("\n") + "\n")
        f.write("\nanalysis:\n")
        for k, v in analysis.items():
            f.write(f"  {k}: {v}\n")


def assert_reference_held_out(
    reference_cell: str, splits: dict, dataset_name: str = "", verbose: bool = True
):
    """
    Fail loudly if the PS-BO reference cell appears in any modelling split.

    splits : {"train": [...], "val": [...], "test": [...]}
             Pass whatever splits the model uses; entries may include or omit
             the .csv extension — both forms are checked.
    """
    if reference_cell is None:
        raise ValueError(f"[{dataset_name}] no reference cell declared.")
    stem = str(reference_cell).replace(".csv", "")
    for split_name, cells in (splits or {}).items():
        members = {str(c).replace(".csv", "") for c in cells or []}
        if stem in members:
            raise AssertionError(
                f"LEAKAGE [{dataset_name}]: PS-BO reference cell '{reference_cell}' is in the '{split_name}' split. The SOC window would be chosen using data the model is fitted on or scored against. Remove it from that split, or choose a different reference cell."
            )
    if verbose:
        names = ", ".join((splits or {}).keys()) or "(no splits declared)"
        print(
            f"[leakage-guard] {dataset_name}: reference '{reference_cell}' is absent from {names}. OK."
        )
    return True


def run_dataset_cli(namespace, runner, allowed_options, argv=None):
    """Run selected experiments, optionally remapping the original path root."""
    import argparse
    from pathlib import Path, PureWindowsPath

    parser = argparse.ArgumentParser(description=namespace["__doc__"])
    parser.add_argument(
        "--run",
        nargs="+",
        type=int,
        choices=allowed_options,
        help="Run selected options without the interactive menu",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Replace the original D:\\Feature Extarction root with this directory",
    )
    args = parser.parse_args(argv)
    if args.root is not None:
        original_root = PureWindowsPath("D:\\Feature Extarction")

        def remap(value):
            if isinstance(value, (str, Path)):
                try:
                    suffix = PureWindowsPath(str(value)).relative_to(original_root)
                except ValueError:
                    return value
                path = args.root.joinpath(*suffix.parts)
                return path if isinstance(value, Path) else str(path)
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, tuple):
                return tuple((remap(item) for item in value))
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return value

        for name, value in list(namespace.items()):
            if name.isupper():
                namespace[name] = remap(value)
    labels = {
        1: "PS-BO window selection + feature extraction (all configured cells)",
        2: "Fixed SOC windows (original configured list)",
        3: "Sampling-frequency sweep",
        4: "Robustness scenarios A-D",
    }
    print(namespace["__doc__"])
    for name in (
        "CALCE_DIR",
        "HNEI_DIR",
        "NAION_DIR",
        "SNL_DIR",
        "STANFORD_DIR",
        "SUMMARY_DIR",
        "DIAG_CAP_DIR",
        "BATCH_DIRS",
        "OUT_DIR",
        "PSBO_ROOT",
        "REFERENCE_CELL",
        "REFERENCE_FILE",
    ):
        if name in namespace:
            print(f"{name}: {namespace[name]}")
    for config in namespace.get("CONFIGS", []):
        print(
            f"{config['name']}: {config['data_dir']} | reference: {config['reference_cell']}"
        )
    print(
        "Fixed windows:",
        ", ".join((f"{lo}–{hi}%" for lo, hi, _ in namespace["FIXED_WINDOWS"])),
    )
    if args.run:
        runner(args.run)
        return
    while True:
        print(
            "\n"
            + "\n".join((f"{option}. {labels[option]}" for option in allowed_options))
        )
        choice = input("Select options (space-separated), or q to quit: ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return
        try:
            selected = [int(item) for item in choice.replace(",", " ").split()]
        except ValueError:
            print("Enter option numbers or q.")
            continue
        if not selected or set(selected) - set(allowed_options):
            print(f"Choose from {allowed_options}.")
            continue
        runner(selected)
