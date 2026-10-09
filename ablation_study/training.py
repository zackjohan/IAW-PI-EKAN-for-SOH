"""PI-EKAN training, evaluation, logging, and checkpoint persistence."""

import copy
import gc
import json
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

if __package__:
    from .pi_ekan import PIEKAN, set_seed
else:
    from pi_ekan import PIEKAN, set_seed
if __package__:
    from .dataloader import load_dataset, save_data_manifest
else:
    from dataloader import load_dataset, save_data_manifest


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


def select_device():
    requested = os.environ.get("PI_EKAN_DEVICE", "auto")
    if requested == "cpu" or not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.empty_cache()
        return "cuda"
    except RuntimeError as exc:
        logging.getLogger(__name__).warning("CUDA unavailable; using CPU: %s", exc)
        return "cpu"


def get_logger(log_path=None, name="research"):
    """Create an isolated run logger, closing any previous handlers."""
    logger = logging.getLogger(
        f'{name}:{Path(log_path).absolute() if log_path else "console"}'
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logger(logger)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


class PIEKANTrainer(PIEKAN):
    """Training and evaluation for PIEKAN; preserves the source loss protocol."""

    def __init__(self, args, seed=None):
        super().__init__(args, seed=seed)
        if args.save_folder and (not os.path.exists(args.save_folder)):
            os.makedirs(args.save_folder)
        log_dir = (
            os.path.join(args.save_folder, args.log_dir)
            if args.save_folder and args.log_dir
            else None
        )
        self.logger = get_logger(log_dir)
        if self.logger:
            self.logger.info(
                f"Training Config: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}"
            )
            self.logger.info(
                f"Loss weighting: adaptive={args.ablation['adaptive']}, bounded={args.ablation['bounded']}, gamma={args.gamma}, sigma_init={args.sigma_init}, sigma_lr={args.sigma_lr}"
            )
            self.logger.info(
                f"Early Stopping Config: Patience={args.early_stop}, Min Delta Ratio={args.early_stop_min_delta_ratio}"
            )
            self.logger.info(
                f"Optimizer Config: solution_u scheduled (warmup_lr={args.warmup_lr}, lr={args.lr}), dynamical_F constant (lr={args.lr_F}), sigma constant (lr={args.sigma_lr})"
            )
        self.data_losses = []
        self.pde_losses = []
        self.monotonicity_losses = []
        self.validation_losses = []
        self.training_losses = []
        self.lambda_data_history = []
        self.lambda_pde_history = []
        self.lambda_mono_history = []
        self.sigma_data_history = []
        self.sigma_pde_history = []
        self.sigma_mono_history = []
        self.best_model = None
        self.optimizer_main = torch.optim.AdamW(
            self.solution_u.parameters(),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
        )
        self.optimizer_aux = torch.optim.AdamW(
            [
                {
                    "params": self.dynamical_F.parameters(),
                    "lr": args.lr_F,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        self.log_sigma_squared_data,
                        self.log_sigma_squared_pde,
                        self.log_sigma_squared_mono,
                    ],
                    "lr": args.sigma_lr,
                    "weight_decay": args.sigma_weight_decay,
                },
            ]
        )
        self.scheduler = PIScheduler(
            optimizer=self.optimizer_main,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr,
            iter_per_epoch=args.iter_per_epoch,
        )
        self.clear_logger = lambda: close_logger(self.logger)
        self.objective_losses = []

    def _get_all_data(self, loader):
        """Helper to cache and preload entire dataset to GPU memory."""
        if loader is None:
            return None
        if not hasattr(loader, "_cached_gpu_data"):
            loader._cached_gpu_data = [
                t.to(self.device) for t in loader.dataset.tensors
            ]
        return loader._cached_gpu_data

    def Train(self, trainloader, validloader=None, testloader=None):
        best_val_mse = float("inf")
        best_epoch = 0
        early_stop_counter = 0
        print("Preloading paired samples to the selected device...")
        train_data = self._get_all_data(trainloader)
        X1_train, X2_train, Y1_train, Y2_train = train_data
        n_train = X1_train.shape[0]
        num_batches = (n_train + self.args.batch_size - 1) // self.args.batch_size
        print("Starting IAW-PI-EKAN training for Battery SOH estimation...")
        print(
            f"Early Stopping: Patience={self.args.early_stop}, Min Delta Ratio={self.args.early_stop_min_delta_ratio}"
        )
        for e in range(1, self.args.epochs + 1):
            epoch_objective = torch.tensor(0.0, device=self.device)
            self.solution_u.train()
            self.dynamical_F.train()
            epoch_loss_data = torch.tensor(0.0, device=self.device)
            epoch_loss_pde = torch.tensor(0.0, device=self.device)
            epoch_loss_mono = torch.tensor(0.0, device=self.device)
            epoch_count = 0
            perm = torch.randperm(n_train, device=self.device)
            for b_idx in range(num_batches):
                start_idx = b_idx * self.args.batch_size
                end_idx = min(start_idx + self.args.batch_size, n_train)
                idx = perm[start_idx:end_idx]
                x1 = X1_train[idx]
                x2 = X2_train[idx]
                y1 = Y1_train[idx]
                y2 = Y2_train[idx]
                u1, f1 = self.compute_pde_residual(x1)
                u2, f2 = self.compute_pde_residual(x2)
                f_target = torch.zeros_like(f1)
                data_loss = 0.5 * F.mse_loss(u1, y1) + 0.5 * F.mse_loss(u2, y2)
                pde_loss = 0.5 * F.mse_loss(f1, f_target) + 0.5 * F.mse_loss(
                    f2, f_target
                )
                mono_loss = self.relu(torch.mul(u2 - u1, y1 - y2)).mean()
                total_loss, lambda_data, lambda_pde, lambda_mono = (
                    self.compute_adaptive_loss(data_loss, pde_loss, mono_loss)
                )
                if self.args.kan_regularization_weight > 0:
                    reg = self.solution_u.regularization_loss(
                        self.args.kan_regularize_activation,
                        self.args.kan_regularize_entropy,
                    ) + self.dynamical_F.regularization_loss(
                        self.args.kan_regularize_activation,
                        self.args.kan_regularize_entropy,
                    )
                    total_loss += self.args.kan_regularization_weight * reg
                self.optimizer_main.zero_grad()
                self.optimizer_aux.zero_grad()
                total_loss.backward()
                if self.args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.args.gradient_clip_norm
                    )
                self.optimizer_main.step()
                self.optimizer_aux.step()
                self.scheduler.step()
                epoch_objective += total_loss.detach() * x1.size(0)
                epoch_loss_data += data_loss.detach() * x1.size(0)
                epoch_loss_pde += pde_loss.detach() * x1.size(0)
                epoch_loss_mono += mono_loss.detach() * x1.size(0)
                epoch_count += x1.size(0)
            avg_data = (epoch_loss_data / epoch_count).item()
            avg_pde = (epoch_loss_pde / epoch_count).item()
            avg_mono = (epoch_loss_mono / epoch_count).item()
            self.objective_losses.append((epoch_objective / epoch_count).item())
            self.data_losses.append(avg_data)
            self.pde_losses.append(avg_pde)
            self.monotonicity_losses.append(avg_mono)
            lam_d, lam_p, lam_m, _, _, _ = self.compute_adaptive_weights()
            training_performance_loss = (
                lam_d * avg_data + lam_p * avg_pde + lam_m * avg_mono
            )
            self.training_losses.append(training_performance_loss.item())
            self.lambda_data_history.append(lam_d.item())
            self.lambda_pde_history.append(lam_p.item())
            self.lambda_mono_history.append(lam_m.item())
            self.sigma_data_history.append(
                torch.sqrt(torch.exp(self.log_sigma_squared_data)).item()
            )
            self.sigma_pde_history.append(
                torch.sqrt(torch.exp(self.log_sigma_squared_pde)).item()
            )
            self.sigma_mono_history.append(
                torch.sqrt(torch.exp(self.log_sigma_squared_mono)).item()
            )
            if self.logger:
                self.logger.info(
                    f"[Train] epoch:{e:4d}, objective:{self.objective_losses[-1]:.8f}, weighted_components:{training_performance_loss.item():.8f}, data_loss:{avg_data:.8f}, pde_loss:{avg_pde:.8f}, mono_loss:{avg_mono:.8f}, λ_data:{lam_d.item():.6f}, λ_pde:{lam_p.item():.6f}, λ_mono:{lam_m.item():.6f}, σ_data:{self.sigma_data_history[-1]:.6f}, σ_pde:{self.sigma_pde_history[-1]:.6f}, σ_mono:{self.sigma_mono_history[-1]:.6f}, lr_main:{self.scheduler.get_lr():.6f}, lr_F:{self.optimizer_aux.param_groups[0]['lr']:.6f}, lr_σ:{self.optimizer_aux.param_groups[1]['lr']:.6f}"
                )
            if e % self.args.validation_frequency == 0 and validloader is not None:
                valid_mse, valid_rel_l2 = self.Valid(validloader)
                if self.logger:
                    self.logger.info(
                        f"[Valid] epoch:{e:4d}, MSE:{valid_mse:.8f}, Rel_L2:{valid_rel_l2:.8f}"
                    )
                    if self.data_losses and self.validation_losses:
                        train_data_loss = self.data_losses[-1]
                        val_mse_rec = self.validation_losses[-1]
                        loss_diff = abs(val_mse_rec - train_data_loss)
                        loss_ratio = (
                            val_mse_rec / train_data_loss
                            if train_data_loss > 1e-08
                            else float("inf")
                        )
                        fit_status, fit_details = pi_detect_overfitting_underfitting(
                            self.data_losses, self.validation_losses
                        )
                        self.logger.info(
                            f"[Loss_Compare] epoch:{e}, Train_Data_Loss:{train_data_loss:.7f}, Val_MSE:{val_mse_rec:.7f}, Diff:{loss_diff:.7f}, Ratio:{loss_ratio:.3f}"
                        )
                        self.logger.info(
                            f"[Fit_Analysis] epoch:{e}, Status:{fit_status}"
                        )
                        if fit_status == "OVERFITTING":
                            self.logger.warning(
                                f"[OVERFITTING_DETECTED] Val MSE ({fit_details['val_loss']:.7f}) > Train data loss ({fit_details['train_loss']:.7f}) | Gap Ratio: {fit_details['loss_gap_ratio']:.3f}"
                            )
                        elif fit_status == "UNDERFITTING":
                            self.logger.warning(
                                f"[UNDERFITTING_DETECTED] Both losses high and not decreasing | Train: {fit_details['train_loss']:.7f}, Val: {fit_details['val_loss']:.7f}"
                            )
                        elif fit_status == "GOOD_FIT":
                            self.logger.info(
                                f"[GOOD_FIT] Losses converging well | Train: {fit_details['train_loss']:.7f}, Val: {fit_details['val_loss']:.7f}"
                            )
            else:
                self.validation_losses.append(
                    self.validation_losses[-1]
                    if self.validation_losses
                    else self.data_losses[-1]
                )
                valid_mse = best_val_mse
            min_delta = (
                self.args.early_stop_min_delta_ratio * best_val_mse
                if best_val_mse < float("inf")
                else 1e-06
            )
            if valid_mse < best_val_mse - min_delta:
                best_val_mse = valid_mse
                best_epoch = e
                early_stop_counter = 0
                if self.logger:
                    self.logger.info(
                        f"[NEW_BEST] Validation MSE: {best_val_mse:.8f} at epoch {e}"
                    )
                if testloader is not None:
                    true_label, pred_label = self.Test(testloader)
                    MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true_label, pred_label)
                    test_rel_l2 = calculate_relative_l2_error(true_label, pred_label)
                    if self.logger:
                        self.logger.info(
                            f"[Test] MSE:{MSE:.8f}, MAE:{MAE:.8f}, MAPE:{MAPE:.4f}%, RMSE:{RMSE:.8f}, R2:{R2:.8f}, Rel_L2:{test_rel_l2:.8f}"
                        )
                    final_fit_status, final_fit_details = (
                        pi_detect_overfitting_underfitting(
                            self.data_losses, self.validation_losses
                        )
                    )
                    lam_d, lam_p, lam_m, _, _, _ = self.compute_adaptive_weights()
                    self.best_model = {
                        "solution_u": copy.deepcopy(self.solution_u.state_dict()),
                        "dynamical_F": copy.deepcopy(self.dynamical_F.state_dict()),
                        "log_sigma_squared_data": self.log_sigma_squared_data.item(),
                        "log_sigma_squared_pde": self.log_sigma_squared_pde.item(),
                        "log_sigma_squared_mono": self.log_sigma_squared_mono.item(),
                        "sigma_data_final": torch.sqrt(
                            torch.exp(self.log_sigma_squared_data)
                        ).item(),
                        "sigma_pde_final": torch.sqrt(
                            torch.exp(self.log_sigma_squared_pde)
                        ).item(),
                        "sigma_mono_final": torch.sqrt(
                            torch.exp(self.log_sigma_squared_mono)
                        ).item(),
                        "lambda_data_final": lam_d.item(),
                        "lambda_pde_final": lam_p.item(),
                        "lambda_mono_final": lam_m.item(),
                        "gamma": self.args.gamma,
                        "epoch": e,
                        "mse": MSE,
                        "mae": MAE,
                        "mape": MAPE,
                        "rmse": RMSE,
                        "r2": R2,
                        "relative_l2_error": test_rel_l2,
                        "data_losses": self.data_losses.copy(),
                        "pde_losses": self.pde_losses.copy(),
                        "monotonicity_losses": self.monotonicity_losses.copy(),
                        "validation_losses": self.validation_losses.copy(),
                        "training_losses": self.training_losses.copy(),
                        "lambda_data_history": self.lambda_data_history.copy(),
                        "lambda_pde_history": self.lambda_pde_history.copy(),
                        "lambda_mono_history": self.lambda_mono_history.copy(),
                        "sigma_data_history": self.sigma_data_history.copy(),
                        "sigma_pde_history": self.sigma_pde_history.copy(),
                        "sigma_mono_history": self.sigma_mono_history.copy(),
                        "fit_status": final_fit_status,
                        "fit_details": final_fit_details,
                    }
                    if self.args.save_folder:
                        sf = self.args.save_folder
                        np.save(os.path.join(sf, "true_label.npy"), true_label)
                        np.save(os.path.join(sf, "pred_label.npy"), pred_label)
                        np.save(
                            os.path.join(sf, "data_losses.npy"),
                            np.array(self.data_losses),
                        )
                        np.save(
                            os.path.join(sf, "pde_losses.npy"),
                            np.array(self.pde_losses),
                        )
                        np.save(
                            os.path.join(sf, "monotonicity_losses.npy"),
                            np.array(self.monotonicity_losses),
                        )
                        np.save(
                            os.path.join(sf, "validation_losses.npy"),
                            np.array(self.validation_losses),
                        )
                        np.save(
                            os.path.join(sf, "training_losses.npy"),
                            np.array(self.training_losses),
                        )
                        np.save(
                            os.path.join(sf, "lambda_data_history.npy"),
                            np.array(self.lambda_data_history),
                        )
                        np.save(
                            os.path.join(sf, "lambda_pde_history.npy"),
                            np.array(self.lambda_pde_history),
                        )
                        np.save(
                            os.path.join(sf, "lambda_mono_history.npy"),
                            np.array(self.lambda_mono_history),
                        )
                        np.save(
                            os.path.join(sf, "sigma_data_history.npy"),
                            np.array(self.sigma_data_history),
                        )
                        np.save(
                            os.path.join(sf, "sigma_pde_history.npy"),
                            np.array(self.sigma_pde_history),
                        )
                        np.save(
                            os.path.join(sf, "sigma_mono_history.npy"),
                            np.array(self.sigma_mono_history),
                        )
                        torch.save(self.best_model, os.path.join(sf, "model.pth"))
            else:
                early_stop_counter += 1
                if self.logger:
                    self.logger.info(
                        f"[NO_IMPROVEMENT] delta < {min_delta:.8f}. Early stop counter: {early_stop_counter}/{self.args.early_stop}"
                    )
            if self.args.early_stop and early_stop_counter >= self.args.early_stop:
                if self.logger:
                    self.logger.info(
                        f"[EARLY_STOP] Stopping at epoch {e} after {early_stop_counter} epochs without improvement (patience={self.args.early_stop})"
                    )
                break
        if self.best_model:
            if self.logger:
                self.logger.info(
                    f"[RESTORE_BEST] Loading best model from epoch {best_epoch}..."
                )
            self.solution_u.load_state_dict(self.best_model["solution_u"])
            self.dynamical_F.load_state_dict(self.best_model["dynamical_F"])
            with torch.no_grad():
                self.log_sigma_squared_data.copy_(
                    torch.tensor(
                        self.best_model["log_sigma_squared_data"], device=self.device
                    )
                )
                self.log_sigma_squared_pde.copy_(
                    torch.tensor(
                        self.best_model["log_sigma_squared_pde"], device=self.device
                    )
                )
                self.log_sigma_squared_mono.copy_(
                    torch.tensor(
                        self.best_model["log_sigma_squared_mono"], device=self.device
                    )
                )
            if self.logger:
                self.logger.info(
                    f"[RESTORE_BEST] Best model fully restored (networks + sigma params). σ_data={self.best_model['sigma_data_final']:.5f}, σ_pde={self.best_model['sigma_pde_final']:.5f}, σ_mono={self.best_model['sigma_mono_final']:.5f}"
                )
        if self.best_model and self.logger:
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(
                f"TRAINING COMPLETED — IAW-PI-EKAN  |  Best Model at Epoch {best_epoch}"
            )
            self.logger.info(f"{'=' * 80}")
            self.logger.info(f"MSE            : {self.best_model['mse']:.8f}")
            self.logger.info(f"MAE            : {self.best_model['mae']:.8f}")
            self.logger.info(f"MAPE           : {self.best_model['mape']:.4f}%")
            self.logger.info(f"RMSE           : {self.best_model['rmse']:.8f}")
            self.logger.info(f"R²             : {self.best_model['r2']:.8f}")
            self.logger.info(
                f"Relative L2    : {self.best_model['relative_l2_error']:.8f}"
            )
            self.logger.info(f"Model Status   : {self.best_model['fit_status']}")
            self.logger.info("\nLOSS WEIGHTS:")
            self.logger.info(
                f"  Bounded weighting: {self.args.ablation['adaptive'] and self.args.ablation['bounded']} | configured gamma: {self.best_model['gamma']}"
            )
            self.logger.info(
                f"  Final λ_data  : {self.best_model['lambda_data_final']:.6f}"
            )
            self.logger.info(
                f"  Final λ_PDE   : {self.best_model['lambda_pde_final']:.6f}"
            )
            self.logger.info(
                f"  Final λ_mono  : {self.best_model['lambda_mono_final']:.6f}"
            )
            self.logger.info(
                f"  Final σ_data  : {self.best_model['sigma_data_final']:.6f}"
            )
            self.logger.info(
                f"  Final σ_PDE   : {self.best_model['sigma_pde_final']:.6f}"
            )
            self.logger.info(
                f"  Final σ_mono  : {self.best_model['sigma_mono_final']:.6f}"
            )
            if self.monotonicity_losses:
                self.logger.info("\nFINAL LOSS COMPONENTS:")
                self.logger.info(f"  Data Loss        : {self.data_losses[-1]:.6f}")
                self.logger.info(f"  PDE Loss         : {self.pde_losses[-1]:.6f}")
                self.logger.info(
                    f"  Monotonicity Loss: {self.monotonicity_losses[-1]:.6f}"
                )
            self.logger.info(f"{'=' * 80}\n")
        self.clear_logger()
        print("\nIAW-PI-EKAN training complete.")

    def Valid(self, validloader):
        """Compute data MSE and relative L2 error on the validation set."""
        self.solution_u.eval()
        self.dynamical_F.eval()
        X1, X2, Y1, Y2 = self._get_all_data(validloader)
        with torch.no_grad():
            u = self.solution_u(X1)
        pred = u.cpu().numpy().flatten()
        true = Y1.cpu().numpy().flatten()
        valid_mse = np.mean((pred - true) ** 2)
        rel_l2 = calculate_relative_l2_error(pred, true)
        self.validation_losses.append(valid_mse)
        return (valid_mse, rel_l2)

    def Test(self, testloader):
        """Return concatenated true and predicted SOH values for the test set."""
        self.solution_u.eval()
        X1, X2, Y1, Y2 = self._get_all_data(testloader)
        with torch.no_grad():
            u = self.solution_u(X1)
        return (Y1.cpu().numpy().flatten(), u.cpu().numpy().flatten())


def run_single_experiment(args, dataset_cfg, experiment_id, output_dir):
    """Run one seed without collecting computational-performance measurements."""
    args = copy.deepcopy(args)
    seed = 42 + experiment_id
    set_seed(seed)
    data = load_dataset(args, dataset_cfg)
    if data["input_dim"] != args.input_dim:
        raise ValueError(
            f"Configured input dimension {args.input_dim} differs from "
            f"the processed data dimension {data['input_dim']}."
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.save_folder = str(output)
    args.log_dir = "log.txt"
    args.iter_per_epoch = data["iter_per_epoch"]
    args.device = select_device()
    save_data_manifest(data, dataset_cfg, output)
    (output / "config.json").write_text(
        json.dumps(dict(vars(args), seed=seed), indent=2), encoding="utf-8"
    )
    trainer = None
    try:
        trainer = PIEKANTrainer(args, seed=seed)
        trainer.Train(data["train"], data["valid"], data["test"])
        if not trainer.best_model:
            raise RuntimeError(
                "No finite improving validation checkpoint was produced."
            )
        best = trainer.best_model
        histories = {
            name: value
            for name, value in vars(trainer).items()
            if isinstance(value, list)
            and (name.endswith("_losses") or name.endswith("_history"))
        }
        for name, values in histories.items():
            np.save(output / f"{name}.npy", np.asarray(values))
        (output / "history.json").write_text(
            json.dumps(histories, indent=2, default=float), encoding="utf-8"
        )
        torch.save(trainer.state_dict(), output / "model_state.pth")
        summary = {
            name: value
            for name, value in best.items()
            if isinstance(value, (str, int, float, bool, np.generic)) or value is None
        }
        summary.update(
            final_data_loss=best["data_losses"][-1],
            final_pde_loss=best["pde_losses"][-1],
            final_mono_loss=best["monotonicity_losses"][-1],
        )
        (output / "results.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )
        (output / "results.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        if trainer is not None:
            close_logger(trainer.logger)
        for split in ("train", "valid", "test"):
            if hasattr(data[split], "_cached_gpu_data"):
                del data[split]._cached_gpu_data
        gc.collect()
