"""One data configuration and preprocessing pipeline for all eight models."""

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

DATASET_CONFIGS = {
    1: {
        "name": "NCM",
        "path": "data/NCM/PS-BO Window Selected",
        "nominal_capacity": 2.0,
        "is_already_normalized": False,
        "train_val_cells": [
            "2C_battery-1",
            "2C_battery-2",
            "2C_battery-3",
            "2C_battery-5",
            "2C_battery-6",
            "2C_battery-7",
            "3C_battery-1",
            "3C_battery-2",
            "3C_battery-3",
            "3C_battery-5",
            "3C_battery-6",
            "3C_battery-7",
            "3C_battery-9",
            "3C_battery-10",
            "3C_battery-11",
            "3C_battery-12",
            "3C_battery-13",
            "3C_battery-14",
            "3C_battery-15",
            "R2.5_battery-1",
            "R2.5_battery-2",
            "R2.5_battery-5",
            "R2.5_battery-6",
            "R2.5_battery-7",
        ],
        "test_cells": [
            "2C_battery-4",
            "2C_battery-8",
            "3C_battery-4",
            "3C_battery-8",
            "R2.5_battery-4",
            "R2.5_battery-8",
        ],
        "num_val_cells": 5,
        "results_tag": "NCM",
    },
    2: {
        "name": "NCA",
        "path": "data/NCA/PS-BO Window Selected",
        "nominal_capacity": 3.5,
        "is_already_normalized": False,
        "train_val_cells": [
            "CY25-05_1-#1",
            "CY25-05_1-#2",
            "CY25-05_1-#3",
            "CY25-05_1-#5",
            "CY25-05_1-#6",
            "CY25-05_1-#7",
            "CY25-05_1-#10",
            "CY25-05_1-#11",
            "CY25-05_1-#12",
            "CY25-05_1-#13",
            "CY25-05_1-#15",
            "CY35-05_1-#2",
            "CY45-05_1-#2",
            "CY45-05_1-#3",
            "CY45-05_1-#5",
            "CY45-05_1-#6",
            "CY45-05_1-#7",
            "CY45-05_1-#9",
            "CY45-05_1-#10",
            "CY45-05_1-#11",
            "CY45-05_1-#12",
            "CY45-05_1-#16",
            "CY45-05_1-#19",
            "CY45-05_1-#20",
        ],
        "test_cells": [
            "CY35-05_1-#1",
            "CY25-05_1-#4",
            "CY45-05_1-#18",
            "CY25-05_1-#14",
            "CY45-05_1-#8",
            "CY45-05_1-#4",
        ],
        "num_val_cells": 5,
        "results_tag": "NCA",
    },
    3: {
        "name": "NCM_NCA",
        "path": "data/NCM_NCA/PS-BO Window Selected",
        "nominal_capacity": 2.5,
        "is_already_normalized": False,
        "train_val_cells": [
            "CY25-05_2-#1",
            "CY25-05_2-#2",
            "CY25-05_1-#3",
            "CY25-05_4-#1",
            "CY25-05_4-#2",
            "CY25-05_4-#3",
        ],
        "test_cells": ["CY25-05_1-#1", "CY25-05_1-#2"],
        "num_val_cells": 1,
        "results_tag": "NCM_NCA",
    },
    4: {
        "name": "LFP",
        "path": "data/LFP/PS-BO Window Selected",
        "nominal_capacity": 1.1,
        "is_already_normalized": False,
        "train_val_cells": [
            "SNL_18650_LFP_15C_0-100_0.5-2C_b",
            "SNL_18650_LFP_25C_0-100_0.5-0.5C_a",
            "SNL_18650_LFP_25C_0-100_0.5-2C_b",
            "SNL_18650_LFP_25C_0-100_0.5-3C_a",
            "SNL_18650_LFP_25C_0-100_0.5-3C_b",
            "SNL_18650_LFP_25C_0-100_0.5-3C_c",
            "SNL_18650_LFP_25C_0-100_0.5-3C_d",
            "SNL_18650_LFP_35C_0-100_0.5-1C_a",
            "SNL_18650_LFP_35C_0-100_0.5-1C_b",
            "SNL_18650_LFP_35C_0-100_0.5-1C_c",
            "SNL_18650_LFP_35C_0-100_0.5-1C_d",
            "SNL_18650_LFP_35C_0-100_0.5-2C_b",
        ],
        "test_cells": [
            "SNL_18650_LFP_25C_0-100_0.5-1C_a",
            "SNL_18650_LFP_25C_0-100_0.5-1C_b",
            "SNL_18650_LFP_25C_0-100_0.5-2C_a",
        ],
        "num_val_cells": 2,
        "results_tag": "LFP",
    },
    5: {
        "name": "LCO",
        "path": "data/LCO/PS-BO Window Selected",
        "nominal_capacity": 1.1,
        "is_already_normalized": False,
        "train_val_cells": [
            "CALCE_CS2_33",
            "CALCE_CS2_35",
            "CALCE_CS2_36",
            "CALCE_CS2_34",
            "CALCE_CS2_38",
            "CALCE_CX2_33",
            "CALCE_CX2_36",
            "CALCE_CX2_34",
        ],
        "test_cells": ["CALCE_CS2_37", "CALCE_CX2_37"],
        "num_val_cells": 2,
        "results_tag": "LCO",
        "cell_nominal_capacities": {
            "CALCE_CX2_33": 1.35,
            "CALCE_CX2_34": 1.35,
            "CALCE_CX2_35": 1.35,
            "CALCE_CX2_36": 1.35,
            "CALCE_CX2_37": 1.35,
        },
    },
    6: {
        "name": "NCM_LCO",
        "path": "data/NCM_LCO/PS-BO Window Selected",
        "nominal_capacity": 2.8,
        "is_already_normalized": False,
        "train_val_cells": [
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_b",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_c",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_d",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_e",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_j",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_l",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_o",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_t",
        ],
        "test_cells": [
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_f",
            "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_g",
        ],
        "num_val_cells": 1,
        "results_tag": "NCM_LCO",
    },
    7: {
        "name": "NA-ion",
        "path": "data/NA-ion/PS-BO Window Selected",
        "nominal_capacity": 1.0,
        "is_already_normalized": False,
        "train_val_cells": [
            "NA-ion_270040-3-8-49",
            "NA-ion_270040-3-5-52",
            "NA-ion_270040-4-5-44",
            "NA-ion_270040-4-7-42",
            "NA-ion_270040-3-6-51",
            "NA-ion_270040-7-3-21",
            "NA-ion_270040-8-2-19",
            "NA-ion_270040-8-4-17",
            "NA-ion_270040-8-6-15",
            "NA-ion_270040-8-7-14",
            "NA-ion_270040-6-2-30",
            "NA-ion_270040-7-2-22",
            "NA-ion_270040-1-5-60",
            "NA-ion_270040-5-7-33",
            "NA-ion_270040-8-1-20",
            "NA-ion_270040-1-2-63",
            "NA-ion_270040-4-1-48",
        ],
        "test_cells": [
            "NA-ion_270040-6-1-31",
            "NA-ion_270040-6-3-29",
            "NA-ion_270040-6-4-28",
            "NA-ion_270040-6-7-25",
        ],
        "num_val_cells": 3,
        "results_tag": "NA-ion",
    },
    8: {
        "name": "Stanford",
        "path": "data/Stanford_NCA/PS-BO Window Selected",
        "nominal_capacity": None,
        "is_already_normalized": True,
        "train_val_cells": [
            "Cell_060",
            "Cell_063",
            "Cell_069",
            "Cell_070",
            "Cell_078",
            "Cell_079",
            "Cell_081",
            "Cell_082",
            "Cell_088",
            "Cell_091",
            "Cell_092",
            "Cell_095",
        ],
        "test_cells": ["Cell_064", "Cell_074", "Cell_096"],
        "num_val_cells": 2,
        "results_tag": "Stanford_NCM",
    },
}


class DataProcessor:
    """
    Handles outlier removal and normalization for multi-cell CSV datasets.

    Normalization statistics are computed once from training cells only and
    then applied unchanged to validation and test cells, preventing data
    leakage across splits.
    """

    def __init__(self, args):
        self.normalization_method = args.normalization_method
        self.args = args
        self.norm_min = None
        self.norm_max = None
        self.norm_mean = None
        self.norm_std = None

    def _3_sigma(self, ser):
        rule = (ser.mean() - 3 * ser.std() > ser) | (ser.mean() + 3 * ser.std() < ser)
        return np.arange(ser.shape[0])[rule]

    def delete_3_sigma(self, df):
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        out_index = set()
        for col in df.columns[:-1]:
            out_index.update(self._3_sigma(df[col]))
        return df.drop(list(out_index)).reset_index(drop=True)

    def fit_normalization_stats(self, feature_df):
        """Compute and store normalization statistics from training cell features."""
        if self.normalization_method == "min-max":
            self.norm_min = feature_df.min()
            self.norm_max = feature_df.max()
        elif self.normalization_method == "z-score":
            self.norm_mean = feature_df.mean()
            self.norm_std = feature_df.std()
            self.norm_std[self.norm_std == 0] = 1.0

    def _apply_normalization(self, f_df):
        """Apply stored training statistics to any feature dataframe."""
        if self.normalization_method == "min-max":
            if self.norm_min is None:
                raise RuntimeError(
                    "fit_normalization_stats() must be called before _apply_normalization()."
                )
            denom = self.norm_max - self.norm_min
            denom[denom == 0] = 1.0
            return 2.0 * (f_df - self.norm_min) / denom - 1.0
        elif self.normalization_method == "z-score":
            if self.norm_mean is None:
                raise RuntimeError(
                    "fit_normalization_stats() must be called before _apply_normalization()."
                )
            return (f_df - self.norm_mean) / self.norm_std
        return f_df

    def process_cell_df_raw(self, cell_df, nominal_capacity, is_already_normalized):
        """
        Raw pre-processing: insert cycle_index, remove feature outliers,
        and normalise the capacity target to SOH.
        Returns a dataframe ready for normalization-stat collection.
        """
        df = cell_df.reset_index(drop=True).copy()
        df.insert(df.shape[1] - 1, "cycle_index", np.arange(df.shape[0], dtype=float))
        df = self.delete_3_sigma(df).reset_index(drop=True)
        target_col = df.columns[-1]
        if not is_already_normalized and nominal_capacity is not None:
            df[target_col] = df[target_col] / nominal_capacity
        return df

    def apply_normalization_to_df(self, df):
        """Apply stored training statistics to a single cell dataframe."""
        f_df = df.iloc[:, :-1].copy()
        f_df_norm = self._apply_normalization(f_df)
        df = df.copy()
        df.iloc[:, :-1] = f_df_norm.astype(float)
        return df

    def process_cell_df(self, cell_df, nominal_capacity, is_already_normalized):
        df = self.process_cell_df_raw(cell_df, nominal_capacity, is_already_normalized)
        if self.norm_min is not None or self.norm_mean is not None:
            df = self.apply_normalization_to_df(df)
        return df

    def create_pairs(self, df):
        """Create consecutive sample pairs (x₁, y₁), (x₂, y₂) from cell data."""
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return ((x[:-1], y[:-1]), (x[1:], y[1:]))


def perform_cell_split(dataset_cfg):
    """
    Deterministic cell-level train / val / test split.

    Test cells are always fixed. The remaining pool is shuffled with a local
    RNG (seed=42) to avoid corrupting global random state, then sliced into
    train and validation subsets.
    """
    all_train_val = list(dataset_cfg["train_val_cells"])
    test_cells = list(dataset_cfg["test_cells"])
    remaining = [c for c in all_train_val if c not in test_cells]
    local_rng = random.Random(42)
    shuffled = remaining.copy()
    local_rng.shuffle(shuffled)
    num_val = dataset_cfg["num_val_cells"]
    train_cells = shuffled[:-num_val] if num_val > 0 else shuffled
    val_cells = shuffled[-num_val:] if num_val > 0 else []
    return (train_cells, val_cells, test_cells)


def create_windowed_sequences(df, window_size):
    """
    Build genuine multi-cycle sequences from a single processed cell
    dataframe (columns = [features..., cycle_index, SOH]).

        X[i] = df.iloc[i : i+window_size, :-1]   -> shape (window_size, input_dim)
        y[i] = df.iloc[i+window_size, -1]        -> next-cycle SOH

    Sequences never cross cell boundaries (this function is called once per
    cell). Returns (None, None) if the cell is too short for one window.
    """
    x_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values.astype(np.float32)
    n = len(df)
    if n < window_size + 1:
        return (None, None)
    X, y = ([], [])
    for i in range(n - window_size):
        X.append(x_all[i : i + window_size])
        y.append(y_all[i + window_size])
    return (np.stack(X, axis=0), np.array(y, dtype=np.float32).reshape(-1, 1))


def get_dataset_config(dataset, data_root="data", dataset_dir=None):
    """Resolve a configured dataset ID/name and a portable CSV directory."""
    import copy

    match = next(
        (
            v
            for k, v in DATASET_CONFIGS.items()
            if str(k) == str(dataset) or v["name"].lower() == str(dataset).lower()
        ),
        None,
    )
    if match is None:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose 1–8 or a configured name."
        )
    config = copy.deepcopy(match)
    relative = Path(config["path"]).relative_to("data")
    config["path"] = str(
        Path(dataset_dir) if dataset_dir else Path(data_root) / relative
    )
    return config


def load_dataset(args, dataset_cfg, small_sample=None, *, mode="pairs", window_size=20):
    """Load shared cells and return paired samples or next-cycle sequences.

    The split uses Random(42); small-sample subsets are prefixes of that split.
    Statistics are fitted before constructing either representation, ensuring
    identical normalization for all models using the same training cells.
    """
    if mode not in {"pairs", "sequence"}:
        raise ValueError("mode must be 'pairs' or 'sequence'")
    if args.batch_size < 1 or window_size < 1:
        raise ValueError("Batch and window sizes must be positive.")
    if small_sample is not None and small_sample < 1:
        raise ValueError("small_sample must be a positive number of cells.")
    folder = Path(dataset_cfg["path"])
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {folder}")
    pool, test = dataset_cfg["train_val_cells"], dataset_cfg["test_cells"]
    if len(set(pool)) != len(pool) or len(set(test)) != len(test):
        raise ValueError("Duplicate cell IDs in dataset configuration.")
    if set(pool) & set(test):
        raise ValueError("Train/validation and test cell pools overlap.")
    train, valid, test = perform_cell_split(dataset_cfg)
    if small_sample is not None:
        if small_sample > len(train):
            raise ValueError(
                f"Requested {small_sample} training cells; only {len(train)} available."
            )
        train = train[:small_sample]
    splits = {"train": train, "valid": valid, "test": test}
    print(f"\nDataset: {dataset_cfg['name']} | directory: {folder}")
    for split, cells in splits.items():
        print(f"{split.upper()} cells ({len(cells)}): {cells}")
    processor = DataProcessor(args)
    files = {p.stem.lower(): p for p in folder.glob("*.csv")}
    raw = {}
    feature_columns = None
    for split, cells in splits.items():
        raw[split] = []
        for cid in cells:
            path = files.get(cid.lower())
            if path is None:
                raise FileNotFoundError(f"Missing configured {split} cell: {cid}.csv")
            df = (
                pd.read_csv(path)
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .reset_index(drop=True)
            )
            if df.shape[1] != 8:
                raise ValueError(
                    f"{cid}: expected seven features followed by capacity/SOH; got {df.shape[1]} columns."
                )
            if feature_columns is None:
                feature_columns = list(df.columns)
            if list(df.columns) != feature_columns:
                raise ValueError(
                    f"{cid}: CSV columns or their ordering differ across cells."
                )
            capacity = dataset_cfg.get("cell_nominal_capacities", {}).get(
                cid, dataset_cfg["nominal_capacity"]
            )
            df = processor.process_cell_df_raw(
                df, capacity, dataset_cfg["is_already_normalized"]
            )
            minimum = 2 if mode == "pairs" else window_size + 1
            if len(df) < minimum:
                raise ValueError(
                    f"{cid}: {len(df)} cleaned rows; {minimum} required for {mode}."
                )
            raw[split].append((cid, df))
        if not raw[split]:
            raise ValueError(f"No cells configured for {split}.")
    training_features = pd.concat(
        [df.iloc[:, :-1] for _, df in raw["train"]], ignore_index=True
    )
    processor.fit_normalization_stats(training_features)
    print(
        f"Normalization: {args.normalization_method}, fitted on {len(training_features)} training rows"
    )
    result = {
        "processor": processor,
        "input_dim": training_features.shape[1],
        "train_cells": train,
        "val_cells": valid,
        "test_cells": test,
        "mode": mode,
        "window_size": window_size if mode == "sequence" else None,
        "sample_cells": {},
        "sample_cycle_indices": {},
    }
    for split, items in raw.items():
        batches = []
        sample_cells, cycle_indices = [], []
        for cid, df in items:
            normalized = processor.apply_normalization_to_df(df)
            if mode == "pairs":
                (x1, y1), (x2, y2) = processor.create_pairs(normalized)
                arrays = (x1, x2, y1.reshape(-1, 1), y2.reshape(-1, 1))
                cycle_indices.extend(df["cycle_index"].iloc[:-1].tolist())
            else:
                arrays = create_windowed_sequences(normalized, window_size)
                cycle_indices.extend(df["cycle_index"].iloc[window_size:].tolist())
            sample_cells.extend([cid] * len(arrays[0]))
            batches.append(arrays)
        tensors = tuple(
            torch.tensor(np.concatenate(parts), dtype=torch.float32)
            for parts in zip(*batches)
        )
        ds = TensorDataset(*tensors)
        result[split] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=split == "train"
        )
        result["sample_cells"][split] = sample_cells
        result["sample_cycle_indices"][split] = cycle_indices
        print(
            f"{split.upper()}: {len(items)} cells, {len(ds)} {mode}, {len(result[split])} batches"
        )
    result["iter_per_epoch"] = len(result["train"])
    result["train_data"] = result["train"].dataset.tensors[0]
    return result


def save_data_manifest(data, config, output_dir):
    """Persist split identities and fitted normalization for reproducibility."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    processor = data["processor"]
    normalization = {"method": processor.normalization_method}
    for name in ("norm_min", "norm_max", "norm_mean", "norm_std"):
        value = getattr(processor, name)
        normalization[name] = value.to_dict() if value is not None else None
    manifest = {
        "dataset": config,
        "train_cells": data["train_cells"],
        "val_cells": data["val_cells"],
        "test_cells": data["test_cells"],
        "mode": data["mode"],
        "window_size": data["window_size"],
        "normalization": normalization,
        "sample_cells": data["sample_cells"],
        "sample_cycle_indices": data["sample_cycle_indices"],
    }
    (path / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (path / "cell_assignment.txt").write_text(
        "\n".join(
            f"{key}: {data[key]}" for key in ("train_cells", "val_cells", "test_cells")
        )
        + "\n",
        encoding="utf-8",
    )
