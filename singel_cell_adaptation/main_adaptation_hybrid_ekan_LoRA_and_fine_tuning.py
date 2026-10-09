"""
Single-cell PI-EKAN adaptation: 16 directed transfer pairs
"""

import argparse
import gc
import glob
import hashlib
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

if __package__:
    from . import config as settings
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from singel_cell_adaptation import config as settings

try:
    from dataloader.dataloader import DataProcessor as SharedDataProcessor
except ModuleNotFoundError as exc:
    if exc.name not in {"dataloader", "dataloader.dataloader"}:
        raise
    raise ModuleNotFoundError(
        "Place singel_cell_adaptation beside the existing PI-EKAN dataloader "
        "folder, then run from the research repository."
    ) from exc

DATASET_CONFIGS = settings.DATASET_CONFIGS
TRANSFER_MAP = settings.TRANSFER_MAP
TRANSFER_PAIRS = settings.TRANSFER_PAIRS
get_args = settings.get_args


def _safe_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        t = torch.zeros(1, device="cuda") + 1
        del t
        torch.cuda.empty_cache()
        return "cuda"
    except Exception as e:
        print(f"CUDA fallback to CPU: {e}")
        return "cpu"


device = _safe_device()
if device == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(log_path=None):
    logger = logging.getLogger("TL-IAW-PI-EKAN")
    close_logger(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_path:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrics(true_label, pred_label):
    MAE = metrics.mean_absolute_error(true_label, pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label, pred_label) * 100
    MSE = metrics.mean_squared_error(true_label, pred_label)
    RMSE = np.sqrt(MSE)
    R2 = metrics.r2_score(true_label, pred_label)
    return (MAE, MAPE, MSE, RMSE, R2)


def rel_l2(pred, true):
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    l2t = np.linalg.norm(true)
    return 0.0 if l2t < 1e-12 else np.linalg.norm(pred - true) / l2t


class ComputationalTracker:

    def __init__(self):
        self.epoch_times = []
        self.gpu_mem = []
        self.cpu_mem = []
        self._train_start = self._epoch_start = None

    def start_training(self):
        self._train_start = time.time()

    def start_epoch(self):
        self._epoch_start = time.time()

    def end_epoch(self):
        t = time.time() - self._epoch_start if self._epoch_start else 0
        self.epoch_times.append(t)
        if torch.cuda.is_available():
            self.gpu_mem.append(
                {
                    "alloc": torch.cuda.memory_allocated() / 1024**3,
                    "peak": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )
        self.cpu_mem.append(psutil.Process().memory_info().rss / 1024**3)
        return t

    def get_training_time(self):
        return time.time() - self._train_start if self._train_start else 0

    def avg_epoch_time(self):
        return float(np.mean(self.epoch_times)) if self.epoch_times else 0

    def memory_stats(self):
        s = {
            "cpu_avg": np.mean(self.cpu_mem) if self.cpu_mem else 0,
            "cpu_peak": np.max(self.cpu_mem) if self.cpu_mem else 0,
        }
        if self.gpu_mem:
            s["gpu_avg"] = np.mean([m["alloc"] for m in self.gpu_mem])
            s["gpu_peak"] = np.max([m["peak"] for m in self.gpu_mem])
        return s


class AdaptationDataProcessor(SharedDataProcessor):
    """Shared preprocessing with the original single-cell transfer protocol."""

    def __init__(self, args):
        super().__init__(args)
        self.input_dim = args.input_dim
        self.method = args.normalization_method
        self.feature_columns = None

    def delete_3_sigma(self, df):
        # Transfer evaluation retains every row, including statistical outliers.
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError(
                "CSV contains missing, nonnumeric or infinite values; fix the data before training."
            )
        return df

    def fit(self, feat_df):
        if self.method not in ("min-max", "z-score"):
            raise ValueError("Unsupported normalization method.")
        self.feature_columns = list(feat_df.columns)
        self.fit_normalization_stats(feat_df)

    def process_raw(self, df, nominal_cap, already_norm):
        if "cycle_index" in df.columns:
            raise ValueError(
                "Input already contains cycle_index; specify its handling explicitly."
            )
        if not already_norm and (nominal_cap is None or nominal_cap <= 0):
            raise ValueError("A positive nominal capacity is required.")
        return self.process_cell_df_raw(df, nominal_cap, already_norm)

    def apply_norm(self, df):
        if list(df.columns[:-1]) != self.feature_columns:
            raise ValueError("Feature names/order differ from fitted scaler.")
        out = self.apply_normalization_to_df(df)
        if not np.isfinite(out.to_numpy()).all():
            raise ValueError("Nonfinite values after normalization.")
        return out

    def create_pairs(self, df, consecutive=None):
        (x1, y1), (x2, y2) = super().create_pairs(df)
        if consecutive is not None:
            return (x1[consecutive], y1[consecutive]), (
                x2[consecutive],
                y2[consecutive],
            )
        return (x1, y1), (x2, y2)


def resolve_folder(path):
    folder = Path(path).expanduser()
    if folder.name == ".csv":
        folder = folder.parent
    return str(folder) if folder.is_dir() else None


def make_rotation_schedule(cfg, seed=42, n_runs=10):
    """Prespecified rule: sort IDs, seeded shuffle once, take the first n_runs IDs.

    Run r adapts on shuffled_pool[r-1]; all remaining cells are its test set.
    Selection uses identifiers only, never features, labels, losses or metrics.
    The number of runs is capped at the pool size: every run needs a distinct
    adaptation cell, so a chemistry with fewer cells than n_runs contributes
    exactly one run per pooled cell.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be positive.")
    pool = list(cfg["cell_pool"])
    keys = [str(cid).strip().casefold() for cid in pool]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate/aliased cell IDs in the target pool.")
    if len(pool) < 2:
        raise ValueError(
            "At least two distinct cells are required (one adaptation, one test)."
        )
    if any((not isinstance(cid, str) or not cid or cid != cid.strip() for cid in pool)):
        raise ValueError(
            "Cell IDs must be nonempty strings without surrounding whitespace."
        )
    pool.sort(key=str.casefold)
    random.Random(seed).shuffle(pool)
    n_eff = min(n_runs, len(pool))
    return [
        {
            "run": r + 1,
            "adapt_cells": [pool[r]],
            "test_cells": [cid for cid in pool if cid != pool[r]],
        }
        for r in range(n_eff)
    ]


def perform_cell_split(cfg, run_id=1, seed=42, n_runs=10):
    schedule = make_rotation_schedule(cfg, seed, n_runs)
    if not 1 <= run_id <= len(schedule):
        raise ValueError(f"run_id must be between 1 and {len(schedule)}.")
    split = schedule[run_id - 1]
    return (list(split["adapt_cells"]), list(split["test_cells"]))


def _load_cells(cell_list, folder, processor, cfg):
    files = {}
    for path in glob.glob(os.path.join(folder, "*.csv")):
        key = os.path.splitext(os.path.basename(path))[0].casefold()
        if key in files:
            raise ValueError(f"Ambiguous CSV filenames: {key}")
        files[key] = path
    raw, records = ([], [])
    for cid in cell_list:
        path = files.get(cid.casefold())
        if path is None:
            raise FileNotFoundError(f"Required cell {cid} missing in {folder}")
        df = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
        if df.shape[1] != processor.input_dim:
            raise ValueError(
                f"{cid}: expected {processor.input_dim - 1} features plus one label column; got {df.shape[1]} columns."
            )
        nc = cfg.get("cell_nominal_capacities", {}).get(cid, cfg["nominal_capacity"])
        proc = processor.process_raw(df, nc, cfg["is_already_normalized"])
        if len(proc) < 2:
            raise ValueError(f"{cid}: fewer than two usable rows.")
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        raw.append((cid, proc, nc))
        records.append(
            {
                "cell": cid,
                "path": os.path.realpath(path),
                "sha256": digest,
                "raw_rows": len(df),
                "usable_rows": len(proc),
                "dropped_nonfinite_rows": len(df) - len(proc),
                "nominal_capacity": nc,
            }
        )
    return (raw, records)


def _raw_to_dataset(raw_dfs, processor, evaluation=False):
    arrays = [[], []] if evaluation else [[], [], [], []]
    schema = None
    for cid, df, _ in raw_dfs:
        if schema is None:
            schema = list(df.columns)
        if list(df.columns) != schema:
            raise ValueError(f"Inconsistent column schema for {cid}")
        normed = processor.apply_norm(df)
        if evaluation:
            values = (normed.iloc[:, :-1].to_numpy(), normed.iloc[:, -1].to_numpy())
        else:
            consecutive = np.diff(df["cycle_index"].to_numpy()) == 1
            if not consecutive.any():
                raise ValueError(f"{cid}: no adjacent valid cycles for training.")
            (x1, y1), (x2, y2) = processor.create_pairs(normed, consecutive)
            values = (x1, x2, y1, y2)
        for dest, value in zip(arrays, values):
            dest.append(value)
    tensors = []
    for chunks in arrays:
        a = np.concatenate(chunks)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        tensors.append(torch.from_numpy(a).float())
    return TensorDataset(*tensors)


def load_target_oneshot(args, target_name, n_shot=1, run_id=1):
    cfg = DATASET_CONFIGS[target_name]
    folder = resolve_folder(cfg["path"])
    if folder is None:
        raise FileNotFoundError(f"Cannot find target folder: {cfg['path']}")
    if n_shot != 1:
        raise ValueError("This protocol uses exactly one labeled target cell per run.")
    n_scheduled = len(make_rotation_schedule(cfg, args.adaptation_seed, args.n_runs))
    adapt_cells, test_cells = perform_cell_split(
        cfg, run_id, args.adaptation_seed, args.n_runs
    )
    print(f"\n{'=' * 60}")
    print(
        f"TARGET DATA: {target_name} ({n_shot}-shot adaptation; rotation {run_id}/{n_scheduled})"
    )
    print(f"  Adaptation cells ({len(adapt_cells)}): {adapt_cells}")
    print(f"  Test cells       ({len(test_cells)}):  {test_cells}")
    print(f"{'=' * 60}")
    processor = AdaptationDataProcessor(args)
    adapt_raw, adapt_records = _load_cells(adapt_cells, folder, processor, cfg)
    test_raw, test_records = _load_cells(test_cells, folder, processor, cfg)
    if list(adapt_raw[0][1].columns) != list(test_raw[0][1].columns):
        raise ValueError("Training and test column schemas differ.")
    if {r["path"] for r in adapt_records} & {r["path"] for r in test_records} or {
        r["sha256"] for r in adapt_records
    } & {r["sha256"] for r in test_records}:
        raise ValueError(
            "Same file or byte-identical data appears in training and test."
        )
    all_adapt_features = pd.concat(
        [df.iloc[:, :-1] for _, df, _ in adapt_raw], ignore_index=True
    )
    processor.fit(all_adapt_features)
    print(
        f"  Normalization fitted on {len(all_adapt_features)} adaptation rows, method='{args.normalization_method}'"
    )
    train_ds = _raw_to_dataset(adapt_raw, processor)
    test_ds = _raw_to_dataset(test_raw, processor, evaluation=True)
    args.iter_per_epoch = max(1, math.ceil(len(train_ds) / args.batch_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    return {
        "train": train_loader,
        "iter_per_epoch": args.iter_per_epoch,
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
        "adapt_cells": adapt_cells,
        "test_cells": test_cells,
        "processor": processor,
        "records": {"training": adapt_records, "testing": test_records},
    }


class Sin(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class KANLinear_Base(nn.Module):

    def __init__(
        self,
        in_f,
        out_f,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_f, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_f, in_f))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_f, in_f, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_f, in_f))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_act = Sin()
        self.grid_eps = grid_eps
        self.enable_ss = True
        self._init()

    def _init(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )
            nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - g[:, : -(k + 1)]) / (g[:, k:-1] - g[:, : -(k + 1)]) * bases[
                :, :, :-1
            ] + (g[:, k + 1 :] - x) / (g[:, k + 1 :] - g[:, 1:-k]) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_out = F.linear(self.base_act(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out

    def reg_loss(self, ra=1.0, re=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        rl = l1.sum()
        p = l1 / (rl + 1e-08)
        return ra * rl + re * -torch.sum(p * p.log().clamp(min=-100))


class EfficientKAN_Base(nn.Module):

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear_Base(
                    i,
                    o,
                    grid_size,
                    spline_order,
                    scale_noise,
                    scale_base,
                    scale_spline,
                    grid_eps,
                    grid_range,
                )
                for i, o in zip(layers_hidden, layers_hidden[1:])
            ]
        )

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def reg_loss(self, ra=1.0, re=1.0):
        return sum((l.reg_loss(ra, re) for l in self.layers))


class KANLinear_LoRA(nn.Module):

    def __init__(
        self,
        in_f,
        out_f,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
        lora_rank=16,
        lora_alpha=32,
    ):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_f, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_f, in_f))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_f, in_f, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_f, in_f))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_act = Sin()
        self.grid_eps = grid_eps
        self.enable_ss = True
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_f))
            self.lora_B = nn.Parameter(torch.zeros(out_f, lora_rank))
            self.scaling = lora_alpha / lora_rank
        self._init()

    def _init(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )
            nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - g[:, : -(k + 1)]) / (g[:, k:-1] - g[:, : -(k + 1)]) * bases[
                :, :, :-1
            ] + (g[:, k + 1 :] - x) / (g[:, k + 1 :] - g[:, 1:-k]) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_x = self.base_act(x)
        base_out = F.linear(base_x, self.base_weight)
        if hasattr(self, "lora_A") and self.lora_rank > 0:
            lora_out = base_x @ self.lora_A.T @ self.lora_B.T * self.scaling
            base_out = base_out + lora_out
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out

    def reg_loss(self, ra=1.0, re=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        rl = l1.sum()
        p = l1 / (rl + 1e-08)
        return ra * rl + re * -torch.sum(p * p.log().clamp(min=-100))


class EfficientKAN_LoRA(nn.Module):

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-1, 1],
        lora_rank=16,
        lora_alpha=32,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear_LoRA(
                    i,
                    o,
                    grid_size,
                    spline_order,
                    scale_noise,
                    scale_base,
                    scale_spline,
                    grid_eps,
                    grid_range,
                    lora_rank,
                    lora_alpha,
                )
                for i, o in zip(layers_hidden, layers_hidden[1:])
            ]
        )

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def reg_loss(self, ra=1.0, re=1.0):
        return sum((l.reg_loss(ra, re) for l in self.layers))

    def configure_for_adaptation(self):
        for name, p in self.named_parameters():
            p.requires_grad = False
            if any(
                (
                    k in name
                    for k in ["lora_A", "lora_B", "spline_weight", "spline_scaler"]
                )
            ):
                p.requires_grad = True

    def trainable_summary(self):
        total = sum((p.numel() for p in self.parameters()))
        train = sum((p.numel() for p in self.parameters() if p.requires_grad))
        return (total, train, 100 * train / total if total else 0)


class LR_Scheduler:

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch=1,
    ):
        warmup_iter = iter_per_epoch * warmup_epochs
        decay_iter = iter_per_epoch * max(1, num_epochs - warmup_epochs)
        self.lr_schedule = np.concatenate(
            [
                np.linspace(warmup_lr, base_lr, warmup_iter),
                final_lr
                + 0.5
                * (base_lr - final_lr)
                * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)),
            ]
        )
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = warmup_lr

    def step(self):
        idx = min(self.iter, len(self.lr_schedule) - 1)
        lr = self.lr_schedule[idx]
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


class _BaseTransferModel(nn.Module):

    def _build_dynamics(self, args):
        f_layers = [args.F_input_dim]
        for _ in range(args.F_layers_num - 1):
            f_layers.append(args.F_hidden_dim)
        f_layers.append(args.F_output_dim)
        return EfficientKAN_Base(
            f_layers,
            args.F_kan_grid_size,
            args.F_kan_spline_order,
            args.F_kan_scale_noise,
            args.F_kan_scale_base,
            args.F_kan_scale_spline,
            args.F_kan_grid_eps,
            args.F_kan_grid_range,
        )

    def _build_sigma(self, args):
        init_val = math.log(args.sigma_init**2)
        self.log_sigma_squared_data = nn.Parameter(
            torch.full((), init_val).to(self.device)
        )
        self.log_sigma_squared_pde = nn.Parameter(
            torch.full((), init_val).to(self.device)
        )
        self.log_sigma_squared_mono = nn.Parameter(
            torch.full((), init_val).to(self.device)
        )
        self.gamma_inv = 1.0 / args.gamma
        self.relu = nn.ReLU()

    def load_pretrained(self, path):
        print(f"\nLoading pretrained model: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        res_u = self.solution_u.load_state_dict(ckpt["solution_u"], strict=False)
        res_f = self.dynamical_F.load_state_dict(ckpt["dynamical_F"], strict=False)
        print(
            f"  solution_u  missing={len(res_u.missing_keys)} unexpected={len(res_u.unexpected_keys)}"
        )
        print(
            f"  dynamical_F missing={len(res_f.missing_keys)} unexpected={len(res_f.unexpected_keys)}"
        )
        bad_missing = [
            k for k in res_u.missing_keys if not k.endswith(("lora_A", "lora_B"))
        ]
        if (
            bad_missing
            or res_u.unexpected_keys
            or res_f.missing_keys
            or res_f.unexpected_keys
        ):
            raise RuntimeError(
                "Pretrained network does not match the requested architecture."
            )
        for key in [
            "log_sigma_squared_data",
            "log_sigma_squared_pde",
            "log_sigma_squared_mono",
        ]:
            if key not in ckpt:
                raise RuntimeError(f"Missing learned checkpoint parameter: {key}")
            if key in ckpt:
                v = ckpt[key]
                val = (
                    torch.tensor(v, device=self.device)
                    if isinstance(v, (int, float))
                    else v.to(self.device)
                )
                getattr(self, key).data.copy_(val)
        if "epoch" in ckpt:
            print(f"  Source epoch: {ckpt['epoch']}")
        if "mse" in ckpt:
            print(f"  Source MSE:   {ckpt['mse']:.8f}")
        for key, value in self.state_dict().items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite pretrained parameter: {key}")
        self._reinit_lora_to_zero()
        print("  Pretrained model loaded.\n")

    def _reinit_lora_to_zero(self):
        for m in self.solution_u.modules():
            if isinstance(m, KANLinear_LoRA) and hasattr(m, "lora_B"):
                nn.init.zeros_(m.lora_B)
                nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))

    def compute_adaptive_weights(self):
        s2d = torch.exp(self.log_sigma_squared_data)
        s2p = torch.exp(self.log_sigma_squared_pde)
        s2m = torch.exp(self.log_sigma_squared_mono)
        return (
            1.0 / (s2d + self.gamma_inv),
            1.0 / (s2p + self.gamma_inv),
            1.0 / (s2m + self.gamma_inv),
            s2d,
            s2p,
            s2m,
        )

    def compute_adaptive_loss(self, dl, pl, ml):
        ld, lp, lm, s2d, s2p, s2m = self.compute_adaptive_weights()
        eps = 1e-08
        total = (
            ld * dl
            + lp * pl
            + lm * ml
            + torch.log(s2d + self.gamma_inv + eps)
            + torch.log(s2p + self.gamma_inv + eps)
            + torch.log(s2m + self.gamma_inv + eps)
        )
        return (total, ld, lp, lm)

    def compute_pde_residual(self, xt):
        xt = xt.detach().requires_grad_(True)
        u = self.solution_u(xt)
        grads = grad(
            u, xt, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_x = grads[:, :-1]
        u_t = grads[:, -1:]
        if u_t is None:
            u_t = torch.zeros_like(xt[:, -1:])
        if u_x is None:
            u_x = torch.zeros_like(xt[:, :-1])
        rhs = torch.cat([xt, u, u_x, u_t], dim=1)
        F_out = self.dynamical_F(rhs)
        return (u, u_t - F_out)

    def forward(self, x):
        return self.solution_u(x)

    def Test(self, loader):
        self.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in loader:
                output = self.solution_u(x.to(self.device))
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Nonfinite test predictions.")
                preds.append(output.cpu().numpy())
                trues.append(y.numpy())
        return (np.concatenate(trues).flatten(), np.concatenate(preds).flatten())

    def _finish_training(self, testloader):
        for key, value in self.state_dict().items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite trained parameter: {key}")
        true, pred = self.Test(testloader)
        MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true, pred)
        result = {
            "solution_u": {
                k: v.detach().cpu().clone()
                for k, v in self.solution_u.state_dict().items()
            },
            "dynamical_F": {
                k: v.detach().cpu().clone()
                for k, v in self.dynamical_F.state_dict().items()
            },
            "log_sigma_squared_data": self.log_sigma_squared_data.item(),
            "log_sigma_squared_pde": self.log_sigma_squared_pde.item(),
            "log_sigma_squared_mono": self.log_sigma_squared_mono.item(),
            "epoch": self.args.adaptation_epochs,
            "mse": MSE,
            "mae": MAE,
            "mape": MAPE,
            "rmse": RMSE,
            "r2": R2,
            "true": true,
            "pred": pred,
            "selection_rule": "final predetermined epoch",
        }
        torch.save(result, os.path.join(self.args.save_folder, "final_model.pth"))
        return result


class StandardFineTuneModel(_BaseTransferModel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = device
        os.makedirs(args.save_folder, exist_ok=True)
        log_path = (
            os.path.join(args.save_folder, args.log_dir) if args.log_dir else None
        )
        self.logger = get_logger(log_path)
        self.solution_u = EfficientKAN_Base(
            args.kan_hidden_layers,
            args.kan_grid_size,
            args.kan_spline_order,
            args.kan_scale_noise,
            args.kan_scale_base,
            args.kan_scale_spline,
            args.kan_grid_eps,
            args.kan_grid_range,
        ).to(device)
        self.dynamical_F = self._build_dynamics(args).to(device)
        self._build_sigma(args)
        self.tracker = ComputationalTracker()

    def configure_for_standard_finetune(self):
        for p in self.dynamical_F.parameters():
            p.requires_grad = False
        for p in self.solution_u.parameters():
            p.requires_grad = True
        args = self.args
        self.log_sigma_squared_data.requires_grad = not args.freeze_adaptive_weights
        self.log_sigma_squared_pde.requires_grad = not args.freeze_adaptive_weights
        self.log_sigma_squared_mono.requires_grad = not args.freeze_adaptive_weights
        total = sum((p.numel() for p in self.parameters()))
        trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        self.logger.info(
            f"FROZEN: dynamical_F  |  FINE-TUNED: solution_u  |  ADAPTIVE: log_sigma (freeze={args.freeze_adaptive_weights})"
        )
        self.logger.info(
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )
        sigma_params = [
            self.log_sigma_squared_data,
            self.log_sigma_squared_pde,
            self.log_sigma_squared_mono,
        ]
        self.optimizer = torch.optim.AdamW(
            [p for p in self.solution_u.parameters() if p.requires_grad],
            lr=self.args.adaptation_lr,
            weight_decay=self.args.weight_decay,
        )
        if not self.args.freeze_adaptive_weights:
            self.sigma_optimizer = torch.optim.AdamW(
                [p for p in sigma_params if p.requires_grad],
                lr=self.args.sigma_lr,
                weight_decay=self.args.sigma_weight_decay,
            )
        else:
            self.sigma_optimizer = None

    def Adaptation(self, trainloader, testloader):
        self.configure_for_standard_finetune()
        self.tracker.start_training()
        if self.args.adaptation_epochs < 1:
            raise ValueError("adaptation_epochs must be positive.")
        for e in range(1, self.args.adaptation_epochs + 1):
            self.train()
            self.dynamical_F.eval()
            for p in self.dynamical_F.parameters():
                p.requires_grad = False
            meter_d = AverageMeter()
            meter_p = AverageMeter()
            meter_m = AverageMeter()
            self.tracker.start_epoch()
            for x1, x2, y1, y2 in trainloader:
                x1, x2, y1, y2 = (
                    x1.to(device),
                    x2.to(device),
                    y1.to(device),
                    y2.to(device),
                )
                u1, f1 = self.compute_pde_residual(x1)
                u2, f2 = self.compute_pde_residual(x2)
                data_loss = 0.5 * F.mse_loss(u1, y1) + 0.5 * F.mse_loss(u2, y2)
                pde_loss = 0.5 * F.mse_loss(
                    f1, torch.zeros_like(f1)
                ) + 0.5 * F.mse_loss(f2, torch.zeros_like(f2))
                mono_loss = self.relu(torch.mul(u2 - u1, y1 - y2)).mean()
                total_loss, _, _, _ = self.compute_adaptive_loss(
                    data_loss, pde_loss, mono_loss
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {e}.")
                self.optimizer.zero_grad()
                if self.sigma_optimizer:
                    self.sigma_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.args.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                if self.sigma_optimizer:
                    self.sigma_optimizer.step()
                n = x1.size(0)
                meter_d.update(data_loss.item(), n)
                meter_p.update(pde_loss.item(), n)
                meter_m.update(mono_loss.item(), n)
            epoch_t = self.tracker.end_epoch()
            if (
                e == 1 or e % 10 == 0 or e == self.args.adaptation_epochs
            ) and self.logger:
                lam = self.compute_adaptive_weights()
                self.logger.info(
                    f"[SFT] e:{e:4d} data:{meter_d.avg:.6f} pde:{meter_p.avg:.6f} mono:{meter_m.avg:.6f} λd:{lam[0].item():.4f} λp:{lam[1].item():.4f} λm:{lam[2].item():.4f} t:{epoch_t:.2f}s"
                )
        train_time = self.tracker.get_training_time()
        best = self._finish_training(testloader)
        mem = self.tracker.memory_stats()
        self.logger.info(
            f"[SFT] Training done. Final epoch:{self.args.adaptation_epochs} Time:{train_time:.1f}s AvgEpoch:{self.tracker.avg_epoch_time():.2f}s CPU_peak:{mem.get('cpu_peak', 0):.2f}GB"
        )
        return best


class HybridKANLoRAModel(_BaseTransferModel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = device
        os.makedirs(args.save_folder, exist_ok=True)
        log_path = (
            os.path.join(args.save_folder, args.log_dir) if args.log_dir else None
        )
        self.logger = get_logger(log_path)
        self.solution_u = EfficientKAN_LoRA(
            args.kan_hidden_layers,
            args.kan_grid_size,
            args.kan_spline_order,
            args.kan_scale_noise,
            args.kan_scale_base,
            args.kan_scale_spline,
            args.kan_grid_eps,
            args.kan_grid_range,
            args.lora_rank,
            args.lora_alpha,
        ).to(device)
        self.dynamical_F = self._build_dynamics(args).to(device)
        self._build_sigma(args)
        self.tracker = ComputationalTracker()

    def configure_for_kanlora(self):
        for p in self.dynamical_F.parameters():
            p.requires_grad = False
        self.solution_u.configure_for_adaptation()
        self.log_sigma_squared_data.requires_grad = False
        self.log_sigma_squared_pde.requires_grad = False
        self.log_sigma_squared_mono.requires_grad = False
        total, train_p, pct = self.solution_u.trainable_summary()
        all_total = sum((p.numel() for p in self.parameters()))
        self.logger.info(
            f"KAN-LoRA Param Efficiency: trainable={train_p:,}/{all_total:,} ({100 * train_p / all_total:.2f}%) [LoRA rank={self.args.lora_rank} alpha={self.args.lora_alpha} no dropout]"
        )
        trainable_params = [p for p in self.solution_u.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.adaptation_lr,
            weight_decay=self.args.weight_decay,
        )

    def Adaptation(self, trainloader, testloader):
        self.configure_for_kanlora()
        self.tracker.start_training()
        if self.args.adaptation_epochs < 1:
            raise ValueError("adaptation_epochs must be positive.")
        for e in range(1, self.args.adaptation_epochs + 1):
            self.train()
            self.dynamical_F.eval()
            for p in self.dynamical_F.parameters():
                p.requires_grad = False
            meter_d = AverageMeter()
            meter_p = AverageMeter()
            meter_m = AverageMeter()
            self.tracker.start_epoch()
            for x1, x2, y1, y2 in trainloader:
                x1, x2, y1, y2 = (
                    x1.to(device),
                    x2.to(device),
                    y1.to(device),
                    y2.to(device),
                )
                u1, f1 = self.compute_pde_residual(x1)
                u2, f2 = self.compute_pde_residual(x2)
                data_loss = 0.5 * F.mse_loss(u1, y1) + 0.5 * F.mse_loss(u2, y2)
                pde_loss = 0.5 * F.mse_loss(
                    f1, torch.zeros_like(f1)
                ) + 0.5 * F.mse_loss(f2, torch.zeros_like(f2))
                mono_loss = self.relu(torch.mul(u2 - u1, y1 - y2)).mean()
                total_loss, _, _, _ = self.compute_adaptive_loss(
                    data_loss, pde_loss, mono_loss
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {e}.")
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.args.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                n = x1.size(0)
                meter_d.update(data_loss.item(), n)
                meter_p.update(pde_loss.item(), n)
                meter_m.update(mono_loss.item(), n)
            epoch_t = self.tracker.end_epoch()
            if (
                e == 1 or e % 10 == 0 or e == self.args.adaptation_epochs
            ) and self.logger:
                lam = self.compute_adaptive_weights()
                lora_norms = [
                    (n, p.norm().item())
                    for n, p in self.named_parameters()
                    if "lora_B" in n and p.requires_grad
                ]
                lora_str = " ".join(
                    (f"{n.split('.')[-2]}:{v:.6f}" for n, v in lora_norms[:3])
                )
                self.logger.info(
                    f"[LoRA] e:{e:4d} data:{meter_d.avg:.6f} pde:{meter_p.avg:.6f} mono:{meter_m.avg:.6f} λd:{lam[0].item():.4f} λp:{lam[1].item():.4f} λm:{lam[2].item():.4f} LoRA_B_norms:[{lora_str}] t:{epoch_t:.2f}s"
                )
        train_time = self.tracker.get_training_time()
        best = self._finish_training(testloader)
        mem = self.tracker.memory_stats()
        total, trainable, pct = self.solution_u.trainable_summary()
        all_total = sum((p.numel() for p in self.parameters()))
        self.logger.info(
            f"[LoRA] Done. Final epoch:{self.args.adaptation_epochs} Time:{train_time:.1f}s AvgEpoch:{self.tracker.avg_epoch_time():.2f}s CPU_peak:{mem.get('cpu_peak', 0):.2f}GB Param_eff={100 * trainable / all_total:.2f}% ({trainable:,}/{all_total:,})"
        )
        return best


def eval_source_only(model, testloader, logger=None):
    true, pred = model.Test(testloader)
    MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true, pred)
    rl2 = rel_l2(pred, true)
    if logger:
        logger.info(
            f"[Source-Only] MAPE:{MAPE:.4f}% RMSE:{RMSE:.8f} MSE:{MSE:.8f} MAE:{MAE:.8f} R2:{R2:.8f} RelL2:{rl2:.8f}"
        )
    return {
        "MAPE": MAPE,
        "RMSE": RMSE,
        "MSE": MSE,
        "MAE": MAE,
        "R2": R2,
        "RelL2": rl2,
        "true": true,
        "pred": pred,
    }


def save_results(
    save_folder,
    source_res,
    finetune_res,
    method_name,
    source_name,
    target_name,
    n_shot,
    run_id=None,
):
    os.makedirs(save_folder, exist_ok=True)
    if run_id is not None:
        summary_path = os.path.join(
            save_folder, f"results_summary_run_{run_id:02d}.txt"
        )
    else:
        summary_path = os.path.join(save_folder, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write("Transfer Learning Results\n")
        if run_id is not None:
            f.write(f"Run: {run_id}\n")
        f.write(f"Method:  {method_name}\n")
        f.write(f"Source:  {source_name}\n")
        f.write(f"Target:  {target_name}\n")
        f.write(f"n-shot:  {n_shot}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(
            "SOURCE-ONLY (source-trained weights; target-adaptation feature scaling):\n"
        )
        f.write(f"  MAPE  : {source_res['MAPE']:.4f}%  [PRIMARY]\n")
        f.write(f"  RMSE  : {source_res['RMSE']:.8f}  [PRIMARY]\n")
        f.write(f"  MSE   : {source_res['MSE']:.8f}\n")
        f.write(f"  MAE   : {source_res['MAE']:.8f}\n")
        f.write(f"  R²    : {source_res['R2']:.8f}\n")
        f.write(f"  RelL2 : {source_res['RelL2']:.8f}\n\n")
        if finetune_res:
            f.write(f"FINE-TUNED ({method_name}):\n")
            f.write(f"  MAPE  : {finetune_res['mape']:.4f}%  [PRIMARY]\n")
            f.write(f"  RMSE  : {finetune_res['rmse']:.8f}  [PRIMARY]\n")
            f.write(f"  MSE   : {finetune_res['mse']:.8f}\n")
            f.write(f"  MAE   : {finetune_res['mae']:.8f}\n")
            f.write(f"  R²    : {finetune_res['r2']:.8f}\n")
            f.write(f"  Final Epoch: {finetune_res['epoch']}\n\n")
            delta_mape = source_res["MAPE"] - finetune_res["mape"]
            delta_rmse = source_res["RMSE"] - finetune_res["rmse"]
            f.write("IMPROVEMENT:\n")
            f.write(f"  ΔMAPE : {delta_mape:+.4f}%\n")
            f.write(f"  ΔRMSE : {delta_rmse:+.8f}\n")
    if run_id is not None:
        prefix = f"run_{run_id:02d}_"
    else:
        prefix = ""
    np.save(os.path.join(save_folder, f"{prefix}source_true.npy"), source_res["true"])
    np.save(os.path.join(save_folder, f"{prefix}source_pred.npy"), source_res["pred"])
    if finetune_res:
        np.save(
            os.path.join(save_folder, f"{prefix}finetuned_true.npy"),
            finetune_res["true"],
        )
        np.save(
            os.path.join(save_folder, f"{prefix}finetuned_pred.npy"),
            finetune_res["pred"],
        )
    print(f"\n  Results saved: {save_folder}")


def save_aggregate_results(
    exp_folder, all_src, all_ft, method, src, tgt, n_shot, n_runs
):
    os.makedirs(exp_folder, exist_ok=True)

    def stats(vals):
        v = np.array(vals)
        return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max())

    src_mapes = [r["MAPE"] for r in all_src]
    src_rmses = [r["RMSE"] for r in all_src]
    src_r2s = [r["R2"] for r in all_src]
    ft_mapes = [r["mape"] for r in all_ft] if all_ft else []
    ft_rmses = [r["rmse"] for r in all_ft] if all_ft else []
    ft_r2s = [r["r2"] for r in all_ft] if all_ft else []
    with open(
        os.path.join(exp_folder, "AGGREGATE_RESULTS.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"{'=' * 70}\nAGGREGATE RESULTS ({n_runs} RUNS)\n")
        f.write(
            f"Method : {method}\nSource : {src} -> Target : {tgt} | n-shot : {n_shot}\n{'=' * 70}\n\n"
        )
        f.write(
            "Baseline uses source-trained weights with target-adaptation feature scaling.\n"
        )
        f.write(
            f"Variability spans {n_runs} distinct adaptation cells and training seeds; overlapping test sets, not independent samples.\n\n"
        )
        f.write(f"{'─' * 70}\nSOURCE-ONLY\n{'─' * 70}\n")
        f.write(f" {'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
        for name, vals in [
            ("MAPE (%)", src_mapes),
            ("RMSE", src_rmses),
            ("R2", src_r2s),
        ]:
            m, s, mn, mx = stats(vals)
            f.write(f" {name:<12} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}\n")
        if ft_mapes:
            f.write(f"\n{'─' * 70}\nFINE-TUNED\n{'─' * 70}\n")
            f.write(
                f" {'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n"
            )
            for name, vals in [
                ("MAPE (%)", ft_mapes),
                ("RMSE", ft_rmses),
                ("R2", ft_r2s),
            ]:
                m, s, mn, mx = stats(vals)
                f.write(f" {name:<12} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}\n")
            delta_mape = np.mean(src_mapes) - np.mean(ft_mapes)
            delta_rmse = np.mean(src_rmses) - np.mean(ft_rmses)
            f.write(
                f"\n MEAN IMPROVEMENT: DELTA_MAPE={delta_mape:+.4f}% DELTA_RMSE={delta_rmse:+.8f}\n"
            )
        f.write(f"\n{'─' * 70}\nPER-RUN TABLE\n{'─' * 70}\n")
        f.write(f" {'Run':>4} {'Src_MAPE':>10} {'Src_RMSE':>12} {'Src_R2':>8}")
        if ft_mapes:
            f.write(f" {'FT_MAPE':>10} {'FT_RMSE':>12} {'FT_R2':>8}")
        f.write("\n")
        for i, sr in enumerate(all_src):
            f.write(
                f" {i + 1:>4} {sr['MAPE']:>10.4f} {sr['RMSE']:>12.8f} {sr['R2']:>8.5f}"
            )
            if ft_mapes:
                fr = all_ft[i]
                f.write(f" {fr['mape']:>10.4f} {fr['rmse']:>12.8f} {fr['r2']:>8.5f}")
            f.write("\n")
    print(f"\n Aggregate results saved: {exp_folder}")


def save_per_cell_metrics(run_dir, data, source_res, finetune_res):
    rows = []
    offset = 0
    for record in data["records"]["testing"]:
        end = offset + record["usable_rows"]
        for label, result in [
            ("Source-Only", source_res),
            ("Fine-Tuned", finetune_res),
        ]:
            true, pred = (result["true"][offset:end], result["pred"][offset:end])
            if len(true) != record["usable_rows"]:
                raise RuntimeError(
                    "Test prediction count differs from the cell manifest."
                )
            mae, mape, mse, rmse, r2 = eval_metrics(true, pred)
            rows.append(
                dict(
                    method=label,
                    cell=record["cell"],
                    n_cycles=len(true),
                    MAE=mae,
                    MAPE=mape,
                    MSE=mse,
                    RMSE=rmse,
                    R2=r2,
                )
            )
        offset = end
    if offset != len(source_res["true"]) or offset != len(finetune_res["true"]):
        raise RuntimeError("Unexpected test predictions outside the split manifest.")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(run_dir, "test_cell_metrics.csv"), index=False)
    table.groupby("method")[["MAE", "MAPE", "MSE", "RMSE", "R2"]].mean().to_csv(
        os.path.join(run_dir, "test_cell_macro_metrics.csv")
    )


def run_experiment(method, source_name, target_name, args):
    if (source_name, target_name) not in TRANSFER_PAIRS:
        raise ValueError(
            f"Transfer pair is not enabled: {source_name} -> {target_name}"
        )
    if method not in METHOD_NAMES.values():
        raise ValueError(f"Unknown adaptation method: {method}")
    if args.n_runs < 1 or args.adaptation_epochs < 1 or args.batch_size < 1:
        raise ValueError("Runs, epochs and batch size must be positive.")
    tag = method.upper().replace(" ", "_").replace("-", "_")
    folder = os.path.join(
        args.results_root,
        "Standard_Fine-Tuning" if "STANDARD" in tag else "Hybrid_KAN-LoRA_Fine-Tuning",
        f"{source_name}_to_{target_name}",
    )
    os.makedirs(folder, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"{method}: {source_name} → {target_name}  ({args.n_shot}-shot)")
    print(f"Output: {folder}")
    print(f"{'=' * 70}")
    set_seed(42)
    pretrain_path = os.path.join(args.pretrained_root, source_name, "model.pth")
    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrain_path}. Train/save the source model first."
        )
    if args.n_shot != 1 or args.n_runs < 1:
        raise ValueError(
            "Use exactly one labeled adaptation cell and a positive number of runs."
        )
    schedule = make_rotation_schedule(
        DATASET_CONFIGS[target_name], args.adaptation_seed, args.n_runs
    )
    with open(
        os.path.join(folder, "rotation_schedule.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(
            f"Rule: sort IDs case-insensitively; shuffle once with random.Random({args.adaptation_seed}); first {len(schedule)} IDs adapt in runs 1..{len(schedule)}; all other cells test.\n"
        )
        f.write(
            f"Requested runs: {args.n_runs}; executed runs: {len(schedule)} (capped at target pool size {len(DATASET_CONFIGS[target_name]['cell_pool'])}).\n"
        )
        f.write(
            "Each run reloads the source checkpoint and fits a fresh adaptation-only scaler.\n"
        )
        f.write(
            "Test sets overlap across runs: reset runs are not statistically independent.\n"
        )
        for split in schedule:
            f.write(
                f"Run {split['run']}: adaptation={split['adapt_cells']}; test={split['test_cells']}\n"
            )
    with open(pretrain_path, "rb") as f:
        source_hash = hashlib.sha256(f.read()).hexdigest()
    exp_start = time.time()
    all_src = []
    all_ft = []
    for run_id in range(1, len(schedule) + 1):
        print(f"\n -- Run {run_id}/{len(schedule)} --")
        set_seed(42 + run_id)
        run_dir = os.path.join(folder, f"Run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)
        args.save_folder = run_dir
        data = load_target_oneshot(args, target_name, n_shot=args.n_shot, run_id=run_id)
        if data is None:
            raise RuntimeError("Target data loading failed.")
        processor = data["processor"]
        scaling = {
            "method": processor.method,
            "feature_columns": processor.feature_columns,
        }
        for key in ("norm_min", "norm_max", "norm_mean", "norm_std"):
            value = getattr(processor, key)
            if value is not None:
                scaling[key] = value.to_numpy()
        np.savez(os.path.join(run_dir, "adaptation_normalization.npz"), **scaling)
        with open(
            os.path.join(run_dir, "split_protocol.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(f"Source: {source_name}; target: {target_name}\n")
            f.write(
                f"Adaptation cells: {data['adapt_cells']}\nTest cells: {data['test_cells']}\n"
            )
            f.write(
                f"Fixed epochs: {args.adaptation_epochs}; runs: {len(schedule)}; seeds: 43 onward\n"
            )
            f.write(
                "One shot = one labeled cell, all usable cycles. Distinct adaptation cell in each run; overlapping test sets.\n"
            )
            f.write(
                "Baseline: source-trained weights with target-adaptation feature scaling.\n"
            )
            f.write(
                "No statistical outlier trimming; nonfinite input stops execution.\n"
            )
            f.write(repr(data["records"]) + "\n")
        with open(
            os.path.join(run_dir, "split_protocol.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(
                f"Run: {run_id}; training seed: {42 + run_id}; split seed: {args.adaptation_seed}\n"
            )
            f.write(f"Source checkpoint SHA256: {source_hash}\n")
        print(f"\n  Adaptation cells : {data['adapt_cells']}")
        print(f"  Test cells       : {data['test_cells']}")
        if "STANDARD" in tag:
            model = StandardFineTuneModel(args).to(device)
        else:
            model = HybridKANLoRAModel(args).to(device)
        with open(pretrain_path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != source_hash:
                raise RuntimeError(
                    "Source checkpoint changed between runs; stop experiment."
                )
        model.load_pretrained(pretrain_path)
        logger = model.logger
        logger.info(f"\n{'─' * 60}")
        logger.info(f"SOURCE-ONLY EVALUATION: {source_name} → {target_name}")
        logger.info(
            "Baseline uses source-trained weights with target-adaptation feature scaling; not strictly target-independent zero-shot evaluation."
        )
        source_res = eval_source_only(model, data["test"], logger)
        print(
            f"\n  [Source-Only] MAPE={source_res['MAPE']:.4f}%  RMSE={source_res['RMSE']:.8f}"
        )
        logger.info(f"\n{'─' * 60}")
        logger.info(f"FINE-TUNING: {method} ({args.n_shot}-shot) on {target_name}")
        ft_start = time.time()
        best = model.Adaptation(data["train"], data["test"])
        ft_time = time.time() - ft_start
        if best:
            print(
                f"\n  [Fine-Tuned]  MAPE={best['mape']:.4f}%  RMSE={best['rmse']:.8f}"
            )
            print(f"  Training time: {ft_time:.1f}s  Final epoch: {best['epoch']}")
            logger.info(f"\nFINAL RESULTS ({method}):")
            logger.info(f"  MAPE  : {best['mape']:.4f}%  [PRIMARY]")
            logger.info(f"  RMSE  : {best['rmse']:.8f}  [PRIMARY]")
            logger.info(f"  MSE   : {best['mse']:.8f}")
            logger.info(f"  MAE   : {best['mae']:.8f}")
            logger.info(f"  R²    : {best['r2']:.8f}")
            logger.info(
                f"  Training time: {ft_time:.1f}s  Final epoch: {best['epoch']}"
            )
            all_ft.append(best)
        else:
            raise RuntimeError("Fine-tuning did not produce a final model.")
        save_per_cell_metrics(run_dir, data, source_res, best)
        all_src.append(source_res)
        save_results(
            run_dir,
            source_res,
            best,
            method,
            source_name,
            target_name,
            args.n_shot,
            run_id=run_id,
        )
        close_logger(model.logger)
        del model, data
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    total_time = time.time() - exp_start
    save_aggregate_results(
        folder,
        all_src,
        all_ft,
        method,
        source_name,
        target_name,
        args.n_shot,
        len(all_src),
    )
    print(f"\n  Total experiment time: {total_time:.1f}s ({total_time / 60:.1f}min)")


def _choose(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt).strip())
            if lo <= v <= hi:
                return v
            print(f"  Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("  Invalid input.")


def print_header():
    print("\n" + "=" * 70)
    print("  IAW-PI-EKAN TRANSFER LEARNING")
    print("  Training / Testing Only | One-Cell Adaptation")
    print("=" * 70)
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")


METHOD_NAMES = {
    "hybrid": "Hybrid KAN-LoRA Fine-Tuning",
    "standard": "Standard Fine-Tuning",
}


def configure_data_root(root):
    previous = settings.DATA_ROOT
    root = Path(root).expanduser().resolve()
    for cfg in DATASET_CONFIGS.values():
        try:
            relative = Path(cfg["path"]).relative_to(previous)
        except ValueError:
            continue
        cfg["path"] = str(root / relative)
    settings.DATA_ROOT = root


def choose_pair():
    sources = list(TRANSFER_MAP)
    print("\nSelect source chemistry:")
    for index, source in enumerate(sources, 1):
        print(f"  {index}. {source} → {', '.join(TRANSFER_MAP[source])}")
    source = sources[_choose("Source: ", 1, len(sources)) - 1]
    targets = TRANSFER_MAP[source]
    print("\nSelect target chemistry:")
    for index, target in enumerate(targets, 1):
        print(f"  {index}. {target}")
    target = targets[_choose("Target: ", 1, len(targets)) - 1]
    return source, target


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-cell adaptation across 16 transfer pairs"
    )
    parser.add_argument("--method", choices=METHOD_NAMES)
    parser.add_argument("--source", choices=TRANSFER_MAP)
    parser.add_argument("--target", choices=["NCM", "NCA", "LFP", "LCO"])
    parser.add_argument(
        "--all-pairs", action="store_true", help="Run all 16 pairs for one method"
    )
    parser.add_argument(
        "--list-pairs", action="store_true", help="List allowed pairs without training"
    )
    parser.add_argument(
        "--data-root", help="Parent of the configured feature directories"
    )
    parser.add_argument(
        "--pretrained-root", help="Parent of <source>/model.pth directories"
    )
    parser.add_argument("--out", help="Output root")
    options = parser.parse_args(argv)
    if options.all_pairs and (options.source or options.target):
        parser.error("--all-pairs cannot be combined with --source or --target")
    if bool(options.source) != bool(options.target):
        parser.error("Provide both --source and --target")
    if options.source and (options.source, options.target) not in TRANSFER_PAIRS:
        parser.error(
            f"Transfer pair is not enabled: {options.source} -> {options.target}"
        )
    if options.list_pairs:
        for index, (source, target) in enumerate(TRANSFER_PAIRS, 1):
            print(f"{index:2d}. {source} -> {target}")
        return
    if options.all_pairs and options.method is None:
        parser.error("Choose --method hybrid or --method standard with --all-pairs")
    args = get_args()
    if options.data_root:
        configure_data_root(options.data_root)
    if options.pretrained_root:
        args.pretrained_root = str(Path(options.pretrained_root).expanduser().resolve())
    if options.out:
        args.results_root = str(Path(options.out).expanduser().resolve())
    print_header()
    print(f"Results: {args.results_root}")
    print(f"Source checkpoints: {args.pretrained_root}")
    print(
        f"Adaptation: {args.n_shot} cell, {args.n_runs} runs, {args.adaptation_epochs} fixed epochs"
    )
    method = options.method
    if method is None:
        print("\n1. Hybrid KAN-LoRA Fine-Tuning\n2. Standard Fine-Tuning")
        method = ["hybrid", "standard"][_choose("Method: ", 1, 2) - 1]
    if options.all_pairs:
        pairs = TRANSFER_PAIRS
    elif options.source:
        pairs = [(options.source, options.target)]
    else:
        pairs = [choose_pair()]
    try:
        for source, target in pairs:
            run_experiment(METHOD_NAMES[method], source, target, args)
    finally:
        close_logger(logging.getLogger("TL-IAW-PI-EKAN"))
    print("\nTransfer learning completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("TL-IAW-PI-EKAN").critical(
            "Experiment stopped.", exc_info=True
        )
        raise SystemExit(1)
