"""Shared numerical utilities and source-specific scheduling conventions."""

import random
import os
import numpy as np
import torch
from sklearn import metrics


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """Accumulates values and computes running mean."""

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
    """Return MAE, MAPE (%), MSE, RMSE, R²."""
    MAE = metrics.mean_absolute_error(true_label, pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label, pred_label) * 100
    MSE = metrics.mean_squared_error(true_label, pred_label)
    RMSE = np.sqrt(MSE)
    R2 = metrics.r2_score(true_label, pred_label)
    return (MAE, MAPE, MSE, RMSE, R2)


def calculate_relative_l2_error(pred_label, true_label):
    """Relative L2 error: ‖pred − true‖₂ / ‖true‖₂."""
    pred = np.array(pred_label).flatten()
    true = np.array(true_label).flatten()
    denom = np.linalg.norm(true)
    return np.linalg.norm(pred - true) / denom if denom > 1e-12 else 0.0


class PIScheduler:
    """Warm-up + cosine decay scheduler applied to a single optimizer."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch=1,
        constant_predictor_lr=False,
    ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        cosine_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup_schedule, cosine_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group.get("name") == "predictor":
                param_group["lr"] = self.base_lr
            else:
                idx = min(self.iter, len(self.lr_schedule) - 1)
                lr = self.lr_schedule[idx]
                param_group["lr"] = lr
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def pi_detect_overfitting_underfitting(train_losses, val_losses, window_size=5):
    """Classify training regime from recent loss history."""
    details = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "loss_gap": 0.0,
        "loss_gap_ratio": 1.0,
        "train_trend": 0.0,
        "val_trend": 0.0,
        "window_size": 0,
        "reason": "",
    }
    if len(train_losses) < 2 or len(val_losses) < 2:
        details["reason"] = "Need at least 2 epochs of data"
        return ("INSUFFICIENT_DATA", details)
    recent_train = (
        train_losses[-window_size:]
        if len(train_losses) >= window_size
        else train_losses
    )
    recent_val = (
        val_losses[-window_size:] if len(val_losses) >= window_size else val_losses
    )
    cur_train = train_losses[-1]
    cur_val = val_losses[-1]
    gap = cur_val - cur_train
    ratio = cur_val / cur_train if cur_train > 1e-08 else float("inf")
    if len(recent_train) >= 3:
        train_trend = (recent_train[-1] - recent_train[0]) / len(recent_train)
        val_trend = (recent_val[-1] - recent_val[0]) / len(recent_val)
    else:
        train_trend = (
            recent_train[-1] - recent_train[0] if len(recent_train) >= 2 else 0.0
        )
        val_trend = recent_val[-1] - recent_val[0] if len(recent_val) >= 2 else 0.0
    details.update(
        {
            "train_loss": cur_train,
            "val_loss": cur_val,
            "loss_gap": gap,
            "loss_gap_ratio": ratio,
            "train_trend": train_trend,
            "val_trend": val_trend,
            "window_size": len(recent_train),
        }
    )
    if (ratio > 1.2 and gap > 0.01) and (val_trend >= 0 and train_trend < 0):
        return ("OVERFITTING", details)
    if ratio > 1.5 and gap > 0.05:
        return ("OVERFITTING", details)
    if (cur_train > 0.1 and cur_val > 0.1) and (
        train_trend >= -0.001 and val_trend >= -0.001
    ):
        return ("UNDERFITTING", details)
    return ("GOOD_FIT", details)


def pinn_detect_overfitting_underfitting(train_losses, val_losses, window_size=5):
    """Classify training regime from recent loss history."""
    details = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "loss_gap": 0.0,
        "loss_gap_ratio": 1.0,
        "train_trend": 0.0,
        "val_trend": 0.0,
        "window_size": 0,
        "reason": "",
    }
    if len(train_losses) < 2 or len(val_losses) < 2:
        details["reason"] = "Need at least 2 epochs of data"
        return ("INSUFFICIENT_DATA", details)
    recent_train = (
        train_losses[-window_size:]
        if len(train_losses) >= window_size
        else train_losses
    )
    recent_val = (
        val_losses[-window_size:] if len(val_losses) >= window_size else val_losses
    )
    cur_train = train_losses[-1]
    cur_val = val_losses[-1]
    gap = cur_val - cur_train
    ratio = cur_val / cur_train if cur_train > 1e-08 else float("inf")
    if len(recent_train) >= 3:
        train_trend = (recent_train[-1] - recent_train[0]) / len(recent_train)
        val_trend = (recent_val[-1] - recent_val[0]) / len(recent_val)
    else:
        train_trend = (
            recent_train[-1] - recent_train[0] if len(recent_train) >= 2 else 0.0
        )
        val_trend = recent_val[-1] - recent_val[0] if len(recent_val) >= 2 else 0.0
    details.update(
        {
            "train_loss": cur_train,
            "val_loss": cur_val,
            "loss_gap": gap,
            "loss_gap_ratio": ratio,
            "train_trend": train_trend,
            "val_trend": val_trend,
            "window_size": len(recent_train),
        }
    )
    if (ratio > 1.2 and gap > 0.01) and (val_trend >= 0 and train_trend < 0):
        return ("OVERFITTING", details)
    elif ratio > 1.5 and gap > 0.05:
        return ("OVERFITTING", details)
    elif (cur_train > 0.1 and cur_val > 0.1) and (
        train_trend >= -0.001 and val_trend >= -0.001
    ):
        return ("UNDERFITTING", details)
    else:
        return ("GOOD_FIT", details)


class EKANScheduler(object):

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch=1,
        constant_predictor_lr=False,
    ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group.get("name") == "predictor":
                param_group["lr"] = self.base_lr
            else:
                idx = min(self.iter, len(self.lr_schedule) - 1)
                lr = param_group["lr"] = self.lr_schedule[idx]
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def ekan_detect_overfitting_underfitting(train_losses, val_losses, window_size=5):
    """Detect overfitting or underfitting from training history."""
    details = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "loss_gap": 0.0,
        "loss_gap_ratio": 1.0,
        "train_trend": 0.0,
        "val_trend": 0.0,
        "window_size": 0,
        "reason": "",
    }
    if len(train_losses) < 2 or len(val_losses) < 2:
        details["reason"] = "Need at least 2 epochs of data"
        return ("INSUFFICIENT_DATA", details)
    recent_train = (
        train_losses[-window_size:]
        if len(train_losses) >= window_size
        else train_losses
    )
    recent_val = (
        val_losses[-window_size:] if len(val_losses) >= window_size else val_losses
    )
    current_train_loss = train_losses[-1]
    current_val_loss = val_losses[-1]
    loss_gap = current_val_loss - current_train_loss
    loss_gap_ratio = (
        current_val_loss / current_train_loss
        if current_train_loss > 1e-08
        else float("inf")
    )
    if len(recent_train) >= 3:
        train_trend = (recent_train[-1] - recent_train[0]) / len(recent_train)
        val_trend = (recent_val[-1] - recent_val[0]) / len(recent_val)
    else:
        train_trend = (
            recent_train[-1] - recent_train[0] if len(recent_train) >= 2 else 0.0
        )
        val_trend = recent_val[-1] - recent_val[0] if len(recent_val) >= 2 else 0.0
    details.update(
        {
            "train_loss": current_train_loss,
            "val_loss": current_val_loss,
            "loss_gap": loss_gap,
            "loss_gap_ratio": loss_gap_ratio,
            "train_trend": train_trend,
            "val_trend": val_trend,
            "window_size": len(recent_train),
        }
    )
    if (loss_gap_ratio > 1.2 and loss_gap > 0.01) and (
        val_trend >= 0 and train_trend < 0
    ):
        return ("OVERFITTING", details)
    elif loss_gap_ratio > 1.5 and loss_gap > 0.05:
        return ("OVERFITTING", details)
    elif (current_train_loss > 0.1 and current_val_loss > 0.1) and (
        train_trend >= -0.001 and val_trend >= -0.001
    ):
        return ("UNDERFITTING", details)
    else:
        return ("GOOD_FIT", details)


class BCScheduler:

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
        self.base_lr = base_lr
        warmup_iter = max(1, iter_per_epoch * warmup_epochs)
        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = max(1, iter_per_epoch * (num_epochs - warmup_epochs))
        cosine_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup_schedule, cosine_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = warmup_lr

    def step(self):
        idx = min(self.iter, len(self.lr_schedule) - 1)
        lr = self.lr_schedule[idx]
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def bc_detect_overfitting_underfitting(train_losses, val_losses, window=5):
    if len(train_losses) < window or len(val_losses) < window:
        return (
            "INSUFFICIENT_DATA",
            {
                "train_loss": train_losses[-1] if train_losses else 0,
                "val_loss": val_losses[-1] if val_losses else 0,
                "loss_gap_ratio": 0.0,
            },
        )
    train_loss = float(np.mean(train_losses[-window:]))
    val_loss = float(np.mean(val_losses[-window:]))
    ratio = val_loss / train_loss if train_loss > 1e-10 else float("inf")
    if ratio > 1.5:
        status = "OVERFITTING"
    elif train_loss > 0.05 and val_loss > 0.05:
        status = "UNDERFITTING"
    else:
        status = "GOOD_FIT"
    return (
        status,
        {"train_loss": train_loss, "val_loss": val_loss, "loss_gap_ratio": ratio},
    )


def sigma_scale_for_target_coverage(
    y, mu, sigma, target_coverage=0.95, z=1.96, eps=1e-08, s_min=1e-06, s_max=None
):
    """Post-hoc sigma scaling for the supplied labels.

    The preserved BC-PINN reporting path supplies TEST labels. Its resulting
    coverage is descriptive, not held-out calibrated coverage."""
    sigma_safe = np.maximum(sigma, eps)
    ratios = np.abs(y - mu) / sigma_safe
    p = np.percentile(ratios, target_coverage * 100.0)
    s = float(p / z)
    s = np.clip(s, s_min, s_max) if s_max is not None else max(s, s_min)
    return s


def coverage_and_width(y, lower, upper):
    cov = float(np.mean((y >= lower) & (y <= upper)))
    width = float(np.mean(upper - lower))
    return (cov, width)


def count_params(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))


PINNScheduler = PIScheduler
SEQScheduler = BCScheduler
seq_detect_overfitting_underfitting = bc_detect_overfitting_underfitting
