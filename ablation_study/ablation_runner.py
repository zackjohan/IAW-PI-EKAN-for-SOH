"""Run the PI-EKAN ablation study interactively or from the command line."""

import argparse
import copy
import math
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from . import ablation_config as settings, pi_ekan
    from .dataloader import DataProcessor
    from .training import PIEKANTrainer, run_single_experiment
else:
    import ablation_config as settings
    import pi_ekan
    from dataloader import DataProcessor
    from training import PIEKANTrainer, run_single_experiment

KAN_BASIS_GRIDSIZE = None
LEGENDRE_DROPOUT = 0.1
BOUND_INPUTS_TANH = True
BASIS_ADD_BASE_BRANCH = False


def _basis_terms(grid_size, spline_order):
    return int(KAN_BASIS_GRIDSIZE or (grid_size + spline_order))


def _coeff_reg(coeff, regularize_activation=1.0, regularize_entropy=1.0):
    a = coeff.abs().mean(dim=-1).flatten()
    s = a.sum()
    p = a / (s + 1e-12)
    return (
        regularize_activation * s
        - regularize_entropy * (p * torch.log(p + 1e-12)).sum()
    )


class _BasisAdapter(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=None,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self._kw = dict(base_activation=base_activation, scale_base=scale_base)
        self.in_features = self.inputdim = in_features
        self.out_features = self.outdim = out_features
        self.n_terms = _basis_terms(grid_size, spline_order)
        self.grid_range = tuple(grid_range)

    def _init_base_branch(self, base_activation, scale_base):
        self.use_base = bool(BASIS_ADD_BASE_BRANCH)
        if not self.use_base:
            return
        self.base_weight = nn.Parameter(
            torch.Tensor(self.out_features, self.in_features)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        self.base_act = base_activation() if base_activation is not None else nn.SiLU()

    def _base(self, x):
        if not getattr(self, "use_base", False):
            return 0.0
        return F.linear(self.base_act(x), self.base_weight)

    def _flatten(self, x):
        out_shape = x.shape[:-1] + (self.outdim,)
        return x.reshape(-1, self.inputdim), out_shape

    def _grid(self):
        return int(torch.clamp(self.gridsize_param, min=1).round().item())


class FourierKANLinear(_BasisAdapter):
    def __init__(self, *a, addbias=True, **kw):
        super().__init__(*a, **kw)
        self.addbias = addbias
        g = self.n_terms
        self.gridsize_param = nn.Parameter(torch.tensor(float(g)))
        self.fouriercoeffs = nn.Parameter(torch.empty(2, self.outdim, self.inputdim, g))
        nn.init.xavier_uniform_(self.fouriercoeffs)
        self._init_base_branch(self._kw["base_activation"], self._kw["scale_base"])
        if addbias:
            self.bias = nn.Parameter(torch.zeros(1, self.outdim))

    def forward(self, x):
        g = self._grid()
        x, out_shape = self._flatten(x)
        if BOUND_INPUTS_TANH:
            x = torch.tanh(x)
        k = torch.arange(1, g + 1, device=x.device, dtype=x.dtype).reshape(1, 1, 1, g)
        xr = x.reshape(x.shape[0], 1, x.shape[1], 1)
        c, s = torch.cos(k * xr), torch.sin(k * xr)
        y = (c * self.fouriercoeffs[0:1, :, :, :g]).sum(dim=(-2, -1))
        y = y + (s * self.fouriercoeffs[1:2, :, :, :g]).sum(dim=(-2, -1))
        y = y + self._base(x)
        if self.addbias:
            y = y + self.bias
        return y.reshape(out_shape)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return _coeff_reg(self.fouriercoeffs, regularize_activation, regularize_entropy)


class WaveletKANLinear(_BasisAdapter):
    def __init__(self, *a, addbias=True, **kw):
        super().__init__(*a, **kw)
        self.addbias = addbias
        g = self.n_terms
        self.gridsize_param = nn.Parameter(torch.tensor(float(g)))
        self.waveletcoeffs = nn.Parameter(torch.empty(2, self.outdim, self.inputdim, g))
        nn.init.xavier_uniform_(self.waveletcoeffs)
        self._init_base_branch(self._kw["base_activation"], self._kw["scale_base"])
        if addbias:
            self.bias = nn.Parameter(torch.zeros(1, self.outdim))

    def forward(self, x):
        g = self._grid()
        x, out_shape = self._flatten(x)
        if BOUND_INPUTS_TANH:
            x = torch.tanh(x)
        scales = torch.linspace(1, g, g, device=x.device, dtype=x.dtype).reshape(
            1, 1, 1, g
        )
        trans = torch.linspace(0, 1, g, device=x.device, dtype=x.dtype).reshape(
            1, 1, 1, g
        )
        xr = x.reshape(x.shape[0], 1, x.shape[1], 1)
        u = (xr - trans) * scales
        env = torch.exp(-(u**2) / 2.0)
        real = torch.cos(math.pi * u) * env
        imag = torch.sin(math.pi * u) * env
        y = (real * self.waveletcoeffs[0:1, :, :, :g]).sum(dim=(-2, -1))
        y = y + (imag * self.waveletcoeffs[1:2, :, :, :g]).sum(dim=(-2, -1))
        y = y + self._base(x)
        if self.addbias:
            y = y + self.bias
        return y.reshape(out_shape)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return _coeff_reg(self.waveletcoeffs, regularize_activation, regularize_entropy)


class LegendreKANLinear(_BasisAdapter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.max_degree = self.n_terms - 1
        self.weights = nn.Parameter(
            torch.randn(self.max_degree + 1, self.inputdim, self.outdim)
        )
        nn.init.orthogonal_(self.weights)
        self.dropout = nn.Dropout(LEGENDRE_DROPOUT)
        self._init_base_branch(self._kw["base_activation"], self._kw["scale_base"])
        self.bias = nn.Parameter(torch.zeros(self.outdim))

    def forward(self, x):
        x, out_shape = self._flatten(x)
        if BOUND_INPUTS_TANH:
            x = torch.tanh(x)
        b = x.shape[0]
        P_nm2 = torch.ones((b, self.inputdim), device=x.device, dtype=x.dtype)
        P_nm1 = x.clone()
        polys = [P_nm2.unsqueeze(-1), P_nm1.unsqueeze(-1)]
        for n in range(2, self.max_degree + 1):
            P_n = ((2 * n - 1) * x * P_nm1 - (n - 1) * P_nm2) / n
            polys.append(P_n.unsqueeze(-1))
            P_nm2, P_nm1 = P_nm1, P_n
        polys = torch.cat(polys, dim=-1)
        polys = self.dropout(polys)
        y = torch.einsum("bif,fio->bo", polys, self.weights) + self.bias
        y = y + self._base(x)
        return y.reshape(out_shape)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return _coeff_reg(
            self.weights.permute(1, 2, 0), regularize_activation, regularize_entropy
        )


BASIS_REGISTRY = {
    "spline": None,
    "fourier": FourierKANLinear,
    "wavelet": WaveletKANLinear,
    "legendre": LegendreKANLinear,
}


class MonoTerm(nn.Module):
    def __init__(self, owner, variant):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.variant = variant

    def forward(self, arg):
        if self.variant == "off":
            return torch.zeros_like(arg)
        if self.variant == "label_aware":
            return torch.relu(arg)
        if self.variant == "label_free":
            cache = getattr(self._owner, "_u_cache", None)
            if cache is None or len(cache) < 2:
                raise RuntimeError(
                    "label_free monotonicity requires two cached predictions; "
                    "compute_pde_residual must be called twice per iteration."
                )
            u1, u2 = cache[-2], cache[-1]
            cache.clear()
            if u1.shape != arg.shape or u2.shape != arg.shape:
                raise RuntimeError(
                    f"monotonicity shape mismatch: u {tuple(u1.shape)} vs "
                    f"arg {tuple(arg.shape)}"
                )
            return torch.relu(u2 - u1)
        raise ValueError(f"unknown monotonicity variant '{self.variant}'")


class Patcher:
    def __init__(self):
        self.orig = {
            "process_cell_df_raw": DataProcessor.process_cell_df_raw,
            "compute_pde_residual": PIEKANTrainer.compute_pde_residual,
            "compute_adaptive_weights": PIEKANTrainer.compute_adaptive_weights,
            "compute_adaptive_loss": PIEKANTrainer.compute_adaptive_loss,
            "model_init": PIEKANTrainer.__init__,
            "KANLinear": pi_ekan.PIKANLinear,
        }

    def _patch_dataprocessor(self, cfg):
        np_ = np

        def process_cell_df_raw(
            inner, cell_df, nominal_capacity, is_already_normalized
        ):
            df = cell_df.reset_index(drop=True).copy()
            target_col_in = df.columns[-1]
            feats = list(df.columns[:-1])

            keep = list(feats) if cfg["use_features"] else []
            if cfg.get("keep_features") is not None:
                keep = [c for c in keep if c in cfg["keep_features"]]
            if cfg.get("drop_features"):
                keep = [c for c in keep if c not in cfg["drop_features"]]
            df = df[keep + [target_col_in]].copy()

            if cfg["use_cycle"]:
                df.insert(
                    df.shape[1] - 1, "cycle_index", np_.arange(df.shape[0], dtype=float)
                )

            df = inner.delete_3_sigma(df).reset_index(drop=True)
            target_col = df.columns[-1]
            if not is_already_normalized and nominal_capacity is not None:
                df[target_col] = df[target_col] / nominal_capacity
            return df

        DataProcessor.process_cell_df_raw = process_cell_df_raw

    def _patch_pde_residual(self, cfg):
        grad = torch.autograd.grad
        g_inputs = cfg["g_inputs"]
        pde_on = cfg["pde"]
        need_cache = cfg["mono"] == "label_free"

        def compute_pde_residual(inner, xt):
            xt = xt.detach().requires_grad_(True)
            u = inner.solution_u(xt)

            if need_cache:
                if not hasattr(inner, "_u_cache"):
                    inner._u_cache = []
                if len(inner._u_cache) >= 2:
                    inner._u_cache.clear()
                inner._u_cache.append(u)

            if not pde_on:
                return u, torch.zeros_like(u)

            all_gradients = grad(
                outputs=u,
                inputs=xt,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]
            u_x = all_gradients[:, :-1]
            u_t = all_gradients[:, -1:]

            if g_inputs == "full":
                z = torch.cat([xt, u, u_x, u_t], dim=1)
            elif g_inputs == "no_ut":
                z = torch.cat([xt, u, u_x], dim=1)
            elif g_inputs == "no_ux":
                z = torch.cat([xt, u, u_t], dim=1)
            elif g_inputs == "t_u":
                z = torch.cat([xt[:, -1:], u], dim=1)
            elif g_inputs == "u_only":
                z = u
            else:
                raise ValueError(f"unknown g_inputs '{g_inputs}'")

            f = u_t - inner.dynamical_F(z)
            return u, f

        PIEKANTrainer.compute_pde_residual = compute_pde_residual

    def _patch_losses(self, cfg):
        adaptive = cfg["adaptive"]
        bounded = cfg["bounded"]
        lam_fixed = cfg["fixed_lambdas"]
        pde_on = cfg["pde"]
        mono_on = cfg["mono"] != "off"

        def compute_adaptive_weights(inner):
            if not adaptive:
                one = torch.ones((), device=inner.device)
                mk = lambda v: torch.as_tensor(
                    float(v), dtype=torch.float32, device=inner.device
                )
                return (
                    mk(lam_fixed[0]),
                    mk(lam_fixed[1]),
                    mk(lam_fixed[2]),
                    one,
                    one,
                    one,
                )
            gi = inner.gamma_inv if bounded else 0.0
            s2d = torch.exp(inner.log_sigma_squared_data)
            s2p = torch.exp(inner.log_sigma_squared_pde)
            s2m = torch.exp(inner.log_sigma_squared_mono)
            return (1.0 / (s2d + gi), 1.0 / (s2p + gi), 1.0 / (s2m + gi), s2d, s2p, s2m)

        def compute_adaptive_loss(inner, data_loss, pde_loss, mono_loss):
            ld, lp, lm, s2d, s2p, s2m = inner.compute_adaptive_weights()
            if not adaptive:
                total = ld * data_loss + lp * pde_loss + lm * mono_loss
                return total, ld, lp, lm
            eps = 1e-8
            gi = inner.gamma_inv if bounded else 0.0
            total = ld * data_loss + torch.log(s2d + gi + eps)
            if pde_on:
                total = total + lp * pde_loss + torch.log(s2p + gi + eps)
            if mono_on:
                total = total + lm * mono_loss + torch.log(s2m + gi + eps)
            return total, ld, lp, lm

        PIEKANTrainer.compute_adaptive_weights = compute_adaptive_weights
        PIEKANTrainer.compute_adaptive_loss = compute_adaptive_loss

    def _patch_model_init(self, cfg):
        orig_init = self.orig["model_init"]
        variant = cfg["mono"]
        freeze_g = cfg.get("freeze_g", False)

        def __init__(inner, args, seed=None):
            orig_init(inner, args, seed=seed)
            inner._u_cache = []
            inner.relu = MonoTerm(inner, variant).to(inner.device)
            if freeze_g:
                for p in inner.dynamical_F.parameters():
                    p.requires_grad_(False)

        PIEKANTrainer.__init__ = __init__

    def _patch_basis(self, cfg):
        cls = BASIS_REGISTRY[cfg["basis"]]
        if cls is not None:
            pi_ekan.PIKANLinear = cls

    def apply(self, cfg):
        self._patch_dataprocessor(cfg)
        self._patch_pde_residual(cfg)
        self._patch_losses(cfg)
        self._patch_model_init(cfg)
        self._patch_basis(cfg)

    def revert(self):
        DataProcessor.process_cell_df_raw = self.orig["process_cell_df_raw"]
        PIEKANTrainer.compute_pde_residual = self.orig["compute_pde_residual"]
        PIEKANTrainer.compute_adaptive_weights = self.orig["compute_adaptive_weights"]
        PIEKANTrainer.compute_adaptive_loss = self.orig["compute_adaptive_loss"]
        PIEKANTrainer.__init__ = self.orig["model_init"]
        pi_ekan.PIKANLinear = self.orig["KANLinear"]


def g_input_dim(g_inputs, D):
    return {
        "full": 2 * D + 1,
        "no_ut": 2 * D,
        "no_ux": D + 2,
        "t_u": 2,
        "u_only": 1,
    }[g_inputs]


def build_args(cfg, n_features_total):
    args = settings.Config()
    args.ablation = copy.deepcopy(cfg)
    n_feat = 0
    if cfg["use_features"]:
        n_feat = n_features_total
        if cfg.get("keep_features") is not None:
            n_feat = len(cfg["keep_features"])
        if cfg.get("drop_features"):
            n_feat -= len(cfg["drop_features"])
    n_cyc = 1 if cfg["use_cycle"] else 0
    D = n_feat + n_cyc
    if D <= 0:
        raise ValueError(f"[{cfg['name']}] input dimension is zero.")

    args.input_dim = D
    args.kan_hidden_layers = [D, 20, 1]
    args.F_input_dim = g_input_dim(cfg["g_inputs"], D)
    if cfg["gamma"] is not None:
        args.gamma = float(cfg["gamma"])
    return args


def validate_cfg(cfg):
    if not cfg["use_cycle"] and cfg["pde"]:
        raise ValueError(
            f"[{cfg['name']}] use_cycle=False with pde=True. Without a cycle "
            f"index there is no temporal coordinate, so du/dt does not exist. "
            f"Set pde=False for this configuration."
        )
    if not cfg["use_features"] and not cfg["use_cycle"]:
        raise ValueError(f"[{cfg['name']}] no inputs selected.")
    if cfg["basis"] not in BASIS_REGISTRY:
        raise ValueError(f"[{cfg['name']}] unknown basis '{cfg['basis']}'")
    if cfg["mono"] not in ("off", "label_aware", "label_free"):
        raise ValueError(f"[{cfg['name']}] unknown mono '{cfg['mono']}'")


FAMILY_INFO = {
    "A": "inputs and feature engineering",
    "B": "dynamics-network inputs",
    "C": "loss components and weighting",
    "D": "SOC window strategy",
    "E": "reference-cell sensitivity",
    "F": "basis function",
}


def _prompt_numbers(prompt, valid):
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  Nothing entered.")
            continue
        if raw.lower() in ("0", "all"):
            return list(valid.keys())
        picks = [p for p in raw.replace(",", " ").split() if p]
        try:
            nums = [int(p) for p in picks]
        except ValueError:
            print("  Enter numbers only (space-separated), or 0 for all.")
            continue
        bad = [n for n in nums if n not in valid]
        if bad:
            print(f"  Not in range: {bad}")
            continue
        return list(dict.fromkeys(nums))


def choose_families(registry):
    print("\n" + "=" * 78)
    print("STEP 1 — FAMILY")
    print("=" * 78)
    keys = sorted(registry.keys())
    numbered = {i + 1: k for i, k in enumerate(keys)}
    for i, k in numbered.items():
        n_cfg = "2 + one per feature" if k == "A" else len(registry[k])
        print(
            f"  {i}. Family {k:1s} — {FAMILY_INFO.get(k, ''):55s} " f"[{n_cfg} configs]"
        )
    print("  0. All families")
    picks = _prompt_numbers("Select family number(s): ", numbered)
    return [numbered[p] for p in picks]


def choose_configs(fam, cfgs):
    print(f"\n{'─' * 78}\nFamily {fam} — {FAMILY_INFO.get(fam, '')}\n{'─' * 78}")
    numbered = {i + 1: c for i, c in enumerate(cfgs)}
    for i, c in numbered.items():
        ds_default = c["datasets"] or settings.ALL_DATASETS
        ds_names = ",".join(settings.DATASET_NAMES.get(d, str(d)) for d in ds_default)
        where = c.get("where", "")
        print(
            f"  {i:2d}. {c['name']:26s} [{where:4s}] {c.get('answers', ''):50s}"
            f"\n        default datasets: {ds_names}"
        )
    print("   0. All configs in this family")
    picks = _prompt_numbers(f"Select config number(s) for family {fam}: ", numbered)
    return [numbered[p] for p in picks]


def detect_feature_names(dataset_key, path_override=None):
    """Read the selected dataset header; never infer features from another cohort."""
    if settings.FEATURE_NAMES is not None:
        names = list(settings.FEATURE_NAMES)
    else:
        folder = Path(path_override or settings.DATASET_CONFIGS[dataset_key]["path"])
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No feature CSVs in {folder}. Set DATA_ROOT or --data-root. "
                "For listing without data, set FEATURE_NAMES in ablation_config.py."
            )
        names = list(pd.read_csv(csvs[0], nrows=0).columns[:-1])
    if len(names) != 7 or len(set(names)) != len(names) or "cycle_index" in names:
        raise ValueError(
            "Expected seven distinct extracted features, without cycle_index."
        )
    return names


def configure_data_root(root):
    """Relocate configured feature paths without changing their relative folders."""
    previous = settings.DATA_ROOT
    root = Path(root).expanduser().resolve()

    def relocate(path):
        try:
            return str(root / Path(path).relative_to(previous))
        except ValueError:
            return path

    for mapping in (settings.WINDOW_FOLDERS, settings.REFERENCE_FOLDERS):
        for folders in mapping.values():
            for label, path in folders.items():
                folders[label] = relocate(path)
    for key, cfg in settings.DATASET_CONFIGS.items():
        cfg["path"] = relocate(cfg["path"])
        settings.PSBO_FEATURE_PATHS[key] = settings.WINDOW_FOLDERS[key]["PS-BO"]
    settings.DATA_ROOT = root


def dataset_keys(cfg, requested=None):
    defaults = cfg["datasets"] or settings.ALL_DATASETS
    if not requested:
        return defaults
    if cfg["path_override"]:
        return [key for key in defaults if key in requested]
    return requested


def run_config(cfg, dataset_key, seeds, out_root, feature_names, dry_run=False):
    validate_cfg(cfg)
    required = set(cfg.get("keep_features") or []) | set(cfg.get("drop_features") or [])
    if missing := required - set(feature_names):
        raise ValueError(
            f"{cfg['name']}: features missing from this dataset: {sorted(missing)}"
        )
    ds_cfg = copy.deepcopy(settings.DATASET_CONFIGS[dataset_key])
    ds_cfg["path"] = cfg["path_override"] or ds_cfg["path"]
    args = build_args(cfg, len(feature_names))
    run_root = Path(out_root) / "runs" / cfg["family"] / cfg["name"]
    args.results_root = str(run_root)
    print(
        f"\n{cfg['name']} | {ds_cfg['name']} | D={args.input_dim} "
        f"F_in={args.F_input_dim} | pde={cfg['pde']} mono={cfg['mono'] != 'off'} "
        f"adaptive={cfg['adaptive']} bounded={cfg['bounded']} "
        f"basis={cfg['basis']} dynamics_inputs={cfg['g_inputs']}"
    )
    if dry_run:
        return []
    rows = []
    patcher = Patcher()
    try:
        patcher.apply(cfg)
        for sid in range(1, seeds + 1):
            output = run_root / ds_cfg["results_tag"] / f"Experiment_{sid}"
            metrics = run_single_experiment(args, ds_cfg, sid, output)
            row = {
                "family": cfg["family"],
                "config": cfg["name"],
                "answers": cfg["answers"],
                "where": cfg["where"],
                "panel": cfg["panel"],
                "dataset": ds_cfg["name"],
                "seed": sid,
                "MAPE": metrics["mape"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
                "final_pde_loss": metrics["final_pde_loss"],
                "final_data_loss": metrics["final_data_loss"],
                "final_mono_loss": metrics["final_mono_loss"],
                "input_dim": args.input_dim,
                "F_input_dim": args.F_input_dim,
                "pde": cfg["pde"],
                "mono": cfg["mono"],
                "adaptive": cfg["adaptive"],
                "bounded": cfg["bounded"],
                "g_inputs": cfg["g_inputs"],
                "basis": cfg["basis"],
                "gamma": args.gamma,
                "freeze_g": cfg["freeze_g"],
            }
            rows.append(row)
            save_summary([row], out_root)
            print(
                f"Seed {sid}: MAPE={metrics['mape']:.4f}% "
                f"RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.4f}"
            )
    finally:
        patcher.revert()
    return rows


def save_summary(rows, out_root):
    """Persist each successful seed and replace duplicate experiment records."""
    folder = Path(out_root) / "csv"
    folder.mkdir(parents=True, exist_ok=True)
    fresh = pd.DataFrame(rows)
    destinations = {"ablation_master.csv": fresh}
    for family, frame in fresh.groupby("family"):
        destinations[f"ablation_{family}.csv"] = frame
    for filename, frame in destinations.items():
        path = folder / filename
        if path.exists():
            frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["family", "config", "dataset", "seed"], keep="last"
        )
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


def execute_plan(plan, options):
    total = sum(
        len(dataset_keys(cfg, options.datasets)) * options.seeds
        for configs in plan.values()
        for cfg in configs
    )
    print(f"\nPlan: {total} training runs | {options.seeds} seeds per configuration")
    print(f"Output: {options.out}")
    failed = []
    for family, configs in plan.items():
        print(f"\nFamily {family} — {FAMILY_INFO[family]}")
        rows = []
        for cfg in configs:
            for key in dataset_keys(cfg, options.datasets):
                try:
                    features = detect_feature_names(key, cfg["path_override"])
                    rows.extend(
                        run_config(
                            cfg,
                            key,
                            options.seeds,
                            options.out,
                            features,
                            options.dry_run,
                        )
                    )
                except Exception:
                    failed.append((cfg["name"], settings.DATASET_NAMES[key]))
                    print(f"[error] {cfg['name']} on {settings.DATASET_NAMES[key]}")
                    traceback.print_exc()
        if rows:
            print(
                pd.DataFrame(rows)
                .groupby(["config", "dataset"])["MAPE"]
                .agg(["mean", "std"])
                .round(4)
                .to_string()
            )
    if failed:
        raise SystemExit(f"{len(failed)} configuration/dataset runs failed: {failed}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PI-EKAN ablation study")
    parser.add_argument("--families", nargs="+", choices=[*FAMILY_INFO, "all"])
    parser.add_argument("--configs", nargs="+", help="Exact configuration names to run")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        choices=settings.ALL_DATASETS,
        help="Override cohorts for A/B/C/F; filter cohorts for D/E",
    )
    parser.add_argument("--seeds", type=int, default=settings.SEEDS)
    parser.add_argument("--out", default=settings.OUT_ROOT)
    parser.add_argument(
        "--data-root", help="Directory containing dataset feature folders"
    )
    parser.add_argument(
        "--list", action="store_true", help="List configurations and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show dimensions without training"
    )
    options = parser.parse_args(argv)
    if options.seeds < 1:
        parser.error("--seeds must be positive")
    if options.data_root:
        configure_data_root(options.data_root)
    interactive = options.families is None and not (
        options.list or options.dry_run or options.configs
    )
    families = options.families or list(FAMILY_INFO)
    if "all" in families:
        families = list(FAMILY_INFO)
    families = list(dict.fromkeys(families))
    if interactive:
        families = choose_families(settings.build_registry([]))
    if "A" in families:
        key = options.datasets[0] if options.datasets else settings.LFP
        feature_names = detect_feature_names(key)
    else:
        feature_names = []
    registry = settings.build_registry(feature_names)
    if interactive:
        plan = {family: choose_configs(family, registry[family]) for family in families}
    else:
        plan = {family: registry[family] for family in families}
    if options.configs:
        known = {cfg["name"] for configs in plan.values() for cfg in configs}
        if unknown := set(options.configs) - known:
            parser.error(
                f"Unknown configuration names in selected families: {sorted(unknown)}"
            )
        plan = {
            family: [cfg for cfg in configs if cfg["name"] in options.configs]
            for family, configs in plan.items()
        }
    if options.list:
        for family, configs in plan.items():
            print(f"\nFamily {family} — {FAMILY_INFO[family]}")
            for cfg in configs:
                keys = dataset_keys(cfg, options.datasets)
                if keys:
                    names = [settings.DATASET_NAMES[key] for key in keys]
                    print(f"  {cfg['name']} | {names} | {cfg['answers']}")
        return
    execute_plan(plan, options)


if __name__ == "__main__":
    main()
