"""Model-specific trainers and experiment execution."""

import os
import gc
import time
import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pi_ekan import PIEKAN
from model.compare_models import PINN, EKAN, BCPINN, build_baseline
from .tracking import ComputationalTracker, get_logger, device
from .util import (
    set_seed,
    AverageMeter,
    eval_metrics,
    calculate_relative_l2_error,
    PIScheduler,
    pi_detect_overfitting_underfitting,
    PINNScheduler,
    pinn_detect_overfitting_underfitting,
    EKANScheduler,
    ekan_detect_overfitting_underfitting,
    BCScheduler,
    bc_detect_overfitting_underfitting,
    SEQScheduler,
    seq_detect_overfitting_underfitting,
    sigma_scale_for_target_coverage,
    coverage_and_width,
    count_params,
)


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
                f"Adaptive Weights Config: γ={args.gamma}, σ_init={args.sigma_init}, σ_lr={args.sigma_lr}"
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
        self.clear_logger = lambda: [
            h.close()
            for h in self.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        self.comp_tracker = ComputationalTracker()
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
        self.comp_tracker.start_training()
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
            self.comp_tracker.start_epoch()
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
            self.comp_tracker.end_epoch()
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
            self.logger.info("\nADAPTIVE LOSS WEIGHTS (IAW-PINN):")
            self.logger.info(f"  Upper bound γ : {self.best_model['gamma']}")
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


class PINNTrainer(PINN):
    """Training and evaluation for PINN; preserves the source loss protocol."""

    def __init__(self, args, seed=None):
        super().__init__(args, seed=seed)
        self.comp_tracker = ComputationalTracker()
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
                f"Constant Weights: α_data={args.alpha_data}, α_pde={args.alpha_pde}, α_mono={args.alpha_mono}"
            )
            self.logger.info(
                f"Early Stopping Config: Patience={args.early_stop}, Min Delta Ratio={args.early_stop_min_delta_ratio}"
            )
            self.logger.info(
                f"Optimizer Config: solution_u scheduled (warmup_lr={args.warmup_lr}, lr={args.lr}), dynamical_F constant (lr={args.lr_F})"
            )
        self.data_losses = []
        self.pde_losses = []
        self.monotonicity_losses = []
        self.validation_losses = []
        self.training_losses = []
        self.best_model = None
        self.optimizer_main = torch.optim.AdamW(
            self.solution_u.parameters(),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
        )
        self.optimizer_aux = torch.optim.AdamW(
            self.dynamical_F.parameters(), lr=args.lr_F, weight_decay=args.weight_decay
        )
        self.scheduler = PINNScheduler(
            optimizer=self.optimizer_main,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr,
            iter_per_epoch=args.iter_per_epoch,
        )
        self.clear_logger = lambda: [
            h.close()
            for h in self.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]

    def Train(self, trainloader, validloader=None, testloader=None):
        """Training loop — PINN-MLP with constant loss weights."""
        self.comp_tracker.start_training()
        best_val_mse = float("inf")
        best_epoch = 0
        early_stop_counter = 0
        print("Starting PINN-MLP training for Battery SOH Estimation")
        print(
            f"Loss Weights: α_data={self.args.alpha_data}, α_pde={self.args.alpha_pde}, α_mono={self.args.alpha_mono}"
        )
        print(
            f"Early Stopping: Patience={self.args.early_stop}, Min Delta Ratio={self.args.early_stop_min_delta_ratio}"
        )
        for e in range(1, self.args.epochs + 1):
            self.solution_u.train()
            self.dynamical_F.train()
            self.comp_tracker.start_epoch()
            self.comp_tracker.reset_memory_tracking()
            meter_data = AverageMeter()
            meter_pde = AverageMeter()
            meter_mono = AverageMeter()
            for i, (x1, x2, y1, y2) in enumerate(trainloader, 1):
                x1, x2, y1, y2 = (t.to(self.device) for t in (x1, x2, y1, y2))
                u1, f1 = self.compute_pde_residual(x1)
                u2, f2 = self.compute_pde_residual(x2)
                f_target = torch.zeros_like(f1)
                data_loss = 0.5 * F.mse_loss(
                    u1, y1, reduction="mean"
                ) + 0.5 * F.mse_loss(u2, y2, reduction="mean")
                pde_loss = 0.5 * F.mse_loss(
                    f1, f_target, reduction="mean"
                ) + 0.5 * F.mse_loss(f2, f_target, reduction="mean")
                monotonicity_loss = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()
                total_loss = self.compute_loss(data_loss, pde_loss, monotonicity_loss)
                self.optimizer_main.zero_grad()
                self.optimizer_aux.zero_grad()
                total_loss.backward()
                if self.args.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.parameters(), self.args.gradient_clip_norm
                    )
                self.optimizer_main.step()
                self.optimizer_aux.step()
                self.scheduler.step()
                n = x1.size(0)
                meter_data.update(data_loss.item(), n)
                meter_pde.update(pde_loss.item(), n)
                meter_mono.update(monotonicity_loss.item(), n)
                if i % self.args.log_frequency == 0 and self.logger:
                    self.logger.info(
                        f"[Train] epoch:{e:4d}, iter:{i:4d}, data_loss:{data_loss.item():.8f}, pde_loss:{pde_loss.item():.8f}, mono_loss:{monotonicity_loss.item():.8f}, lr_main:{self.scheduler.get_lr():.6f}, lr_F:{self.optimizer_aux.param_groups[0]['lr']:.6f}"
                    )
            self.data_losses.append(meter_data.avg)
            self.pde_losses.append(meter_pde.avg)
            self.monotonicity_losses.append(meter_mono.avg)
            epoch_train_loss = (
                self.args.alpha_data * meter_data.avg
                + self.args.alpha_pde * meter_pde.avg
                + self.args.alpha_mono * meter_mono.avg
            )
            self.training_losses.append(epoch_train_loss)
            epoch_time = self.comp_tracker.end_epoch()
            avg_ep_time = self.comp_tracker.get_average_epoch_time()
            est_total = self.comp_tracker.get_total_training_time_estimate(
                self.args.epochs
            )
            train_elapsed = self.comp_tracker.get_training_time()
            if self.logger:
                self.logger.info(
                    f"[Train] epoch:{e:4d}, data_loss:{meter_data.avg:.8f}, pde_loss:{meter_pde.avg:.8f}, mono_loss:{meter_mono.avg:.8f}, total_loss:{epoch_train_loss:.8f}, lr_main:{self.scheduler.get_lr():.6f}, lr_F:{self.optimizer_aux.param_groups[0]['lr']:.6f}"
                )
                self.logger.info(
                    f"[Timing] epoch_time:{epoch_time:.2f}s, avg_epoch_time:{avg_ep_time:.2f}s, training_time:{train_elapsed:.1f}s, estimated_total:{est_total:.1f}s"
                )
            if e % self.args.validation_frequency == 0 and validloader is not None:
                valid_mse, valid_rel_l2 = self.Valid(validloader)
                if self.logger:
                    self.logger.info(
                        f"[Valid] epoch:{e:4d}, MSE:{valid_mse:.8f}, Rel_L2:{valid_rel_l2:.8f}"
                    )
                    if self.data_losses and self.validation_losses:
                        train_data_loss = self.data_losses[-1]
                        val_mse_last = self.validation_losses[-1]
                        loss_diff = abs(val_mse_last - train_data_loss)
                        loss_ratio = (
                            val_mse_last / train_data_loss
                            if train_data_loss > 1e-08
                            else float("inf")
                        )
                        fit_status, fit_details = pinn_detect_overfitting_underfitting(
                            self.data_losses, self.validation_losses
                        )
                        self.logger.info(
                            f"[Loss_Compare] epoch:{e}, Train_Data_Loss:{train_data_loss:.7f}, Val_MSE:{val_mse_last:.7f}, Diff:{loss_diff:.7f}, Ratio:{loss_ratio:.3f}"
                        )
                        self.logger.info(
                            f"[Fit_Analysis] epoch:{e}, Status:{fit_status}"
                        )
                        if fit_status == "OVERFITTING":
                            self.logger.warning(
                                f"[OVERFITTING_DETECTED] Val MSE ({fit_details['val_loss']:.7f}) > Train Loss ({fit_details['train_loss']:.7f}) | Gap Ratio: {fit_details['loss_gap_ratio']:.3f}"
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
                        f"[NEW_BEST] New best validation MSE: {best_val_mse:.8f} at epoch {e} (improvement threshold: {min_delta:.8f})"
                    )
                if testloader is not None:
                    true_label, pred_label = self.Test(testloader)
                    MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true_label, pred_label)
                    test_rel_l2 = calculate_relative_l2_error(pred_label, true_label)
                    if self.logger:
                        self.logger.info(
                            f"[Test] MSE:{MSE:.8f}, MAE:{MAE:.8f}, MAPE:{MAPE:.4f}%, RMSE:{RMSE:.8f}, R2:{R2:.8f}, Rel_L2:{test_rel_l2:.8f}"
                        )
                    final_fit_status, final_fit_details = (
                        pinn_detect_overfitting_underfitting(
                            self.data_losses, self.validation_losses
                        )
                    )
                    memory_stats = self.comp_tracker.get_memory_stats()
                    self.best_model = {
                        "solution_u": copy.deepcopy(self.solution_u.state_dict()),
                        "dynamical_F": copy.deepcopy(self.dynamical_F.state_dict()),
                        "alpha_data": self.args.alpha_data,
                        "alpha_pde": self.args.alpha_pde,
                        "alpha_mono": self.args.alpha_mono,
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
                        "fit_status": final_fit_status,
                        "fit_details": final_fit_details,
                        "epoch_times": self.comp_tracker.epoch_times.copy(),
                        "average_epoch_time": self.comp_tracker.get_average_epoch_time(),
                        "total_training_time": self.comp_tracker.get_training_time(),
                        "memory_stats": memory_stats,
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
                            os.path.join(sf, "epoch_times.npy"),
                            np.array(self.comp_tracker.epoch_times),
                        )
                        np.save(
                            os.path.join(sf, "cpu_memory_usage.npy"),
                            np.array(self.comp_tracker.cpu_memory_usage),
                        )
                        if self.comp_tracker.gpu_memory_usage:
                            np.save(
                                os.path.join(sf, "gpu_memory_allocated.npy"),
                                np.array(
                                    [
                                        m["allocated"]
                                        for m in self.comp_tracker.gpu_memory_usage
                                    ]
                                ),
                            )
                            np.save(
                                os.path.join(sf, "gpu_memory_max_allocated.npy"),
                                np.array(
                                    [
                                        m["max_allocated"]
                                        for m in self.comp_tracker.gpu_memory_usage
                                    ]
                                ),
                            )
                            np.save(
                                os.path.join(sf, "gpu_memory_reserved.npy"),
                                np.array(
                                    [
                                        m["reserved"]
                                        for m in self.comp_tracker.gpu_memory_usage
                                    ]
                                ),
                            )
                        torch.save(self.best_model, os.path.join(sf, "model.pth"))
            else:
                early_stop_counter += 1
                if self.logger:
                    self.logger.info(
                        f"[NO_IMPROVEMENT] No significant improvement (delta < {min_delta:.8f}). Early stop counter: {early_stop_counter}/{self.args.early_stop}"
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
            if self.logger:
                self.logger.info("[RESTORE_BEST] Best model restored successfully.")
        if testloader is not None:
            print(
                "\n[Inference Benchmark] Measuring prediction speed (solution_u only)"
            )
            self.solution_u.eval()
            sample_batch = next(iter(testloader))
            x_sample = sample_batch[0].to(self.device)
            with torch.no_grad():
                for _ in range(5):
                    _ = self.solution_u(x_sample)
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize()
            NUM_RUNS = 50
            times = []
            with torch.no_grad():
                for _ in range(NUM_RUNS):
                    t0 = time.perf_counter()
                    _ = self.solution_u(x_sample)
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
            avg_time_sec = sum(times) / len(times)
            n_samples = x_sample.shape[0]
            ms_per_1000 = avg_time_sec / n_samples * 1000000
            print(f"→ Batch size used            : {n_samples}")
            print(f"→ Average forward pass time  : {avg_time_sec * 1000:.3f} ms")
            print(
                f"→ Inference speed            : {ms_per_1000:.3f} ms per 1000 samples"
            )
            if self.best_model is not None:
                self.best_model["inference_time_ms_per_1000"] = ms_per_1000
                self.best_model["inference_batch_size_measured"] = n_samples
                self.best_model["inference_num_timing_runs"] = NUM_RUNS
                self.best_model["inference_avg_pass_ms"] = avg_time_sec * 1000
        final_training_time = self.comp_tracker.get_training_time()
        final_avg_epoch_time = self.comp_tracker.get_average_epoch_time()
        final_memory_stats = self.comp_tracker.get_memory_stats()
        if self.best_model and self.logger:
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(
                f"TRAINING COMPLETED — PINN-MLP  |  Best Model at Epoch {best_epoch}"
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
            self.logger.info("\nCONSTANT LOSS WEIGHTS:")
            self.logger.info(f"  α_data : {self.best_model['alpha_data']}")
            self.logger.info(f"  α_PDE  : {self.best_model['alpha_pde']}")
            self.logger.info(f"  α_mono : {self.best_model['alpha_mono']}")
            self.logger.info("\nFINAL LOSS COMPONENTS:")
            self.logger.info(f"  Data Loss        : {self.data_losses[-1]:.6f}")
            self.logger.info(f"  PDE Loss         : {self.pde_losses[-1]:.6f}")
            self.logger.info(f"  Monotonicity Loss: {self.monotonicity_losses[-1]:.6f}")
            self.logger.info("\nCOMPUTATIONAL PERFORMANCE:")
            self.logger.info(
                f"  Total Training Time  : {final_training_time:.1f}s ({final_training_time / 60:.1f}min)"
            )
            self.logger.info(f"  Average Epoch Time   : {final_avg_epoch_time:.2f}s")
            self.logger.info(
                f"  Total Epochs Trained : {len(self.comp_tracker.epoch_times)}"
            )
            self.logger.info("\nMEMORY USAGE:")
            self.logger.info(
                f"  CPU — Avg: {final_memory_stats['cpu_memory_avg']:.2f}GB, Peak: {final_memory_stats['cpu_memory_max']:.2f}GB"
            )
            if "gpu_memory_avg_allocated" in final_memory_stats:
                self.logger.info(
                    f"  GPU — Avg Allocated: {final_memory_stats['gpu_memory_avg_allocated']:.2f}GB, Peak Allocated: {final_memory_stats['gpu_memory_max_allocated']:.2f}GB"
                )
            if self.best_model.get("inference_time_ms_per_1000") is not None:
                self.logger.info("\nINFERENCE PERFORMANCE (solution_u only):")
                self.logger.info(
                    f"  Inference speed  : {self.best_model['inference_time_ms_per_1000']:.3f} ms per 1000 samples"
                )
                self.logger.info(
                    f"  Avg forward pass : {self.best_model['inference_avg_pass_ms']:.3f} ms"
                )
            self.logger.info(f"{'=' * 80}\n")
        self.clear_logger()
        print("\nPINN-MLP training complete — Constant Loss Weights.")
        print(
            f"Loss Weights: Data={self.args.alpha_data}, PDE={self.args.alpha_pde}, Monotonic={self.args.alpha_mono}"
        )

    def Valid(self, validloader):
        """Compute data MSE and relative L2 error on the validation set."""
        self.solution_u.eval()
        self.dynamical_F.eval()
        all_pred, all_true = ([], [])
        with torch.no_grad():
            for x1, x2, y1, y2 in validloader:
                u = self.solution_u(x1.to(self.device))
                all_pred.append(u.cpu().numpy())
                all_true.append(y1.numpy())
        pred = np.concatenate(all_pred).flatten()
        true = np.concatenate(all_true).flatten()
        valid_mse = np.mean((pred - true) ** 2)
        rel_l2 = calculate_relative_l2_error(pred, true)
        self.validation_losses.append(valid_mse)
        return (valid_mse, rel_l2)

    def Test(self, testloader):
        """Return concatenated true and predicted SOH values for the test set."""
        self.solution_u.eval()
        all_pred, all_true = ([], [])
        with torch.no_grad():
            for x1, x2, y1, y2 in testloader:
                u = self.solution_u(x1.to(self.device))
                all_pred.append(u.cpu().numpy())
                all_true.append(y1.numpy())
        return (np.concatenate(all_true).flatten(), np.concatenate(all_pred).flatten())


class EKANTrainer(EKAN):
    """Training and evaluation for EKAN; preserves the source loss protocol."""

    def __init__(self, args, seed=None):
        super().__init__(args, seed=seed)
        self.comp_tracker = ComputationalTracker()
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
                f"Training Config: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, warmup_lr={args.warmup_lr}, final_lr={args.final_lr}"
            )
            self.logger.info(
                f"Early Stopping Config: Patience={args.early_stop}, Min Delta Ratio={args.early_stop_min_delta_ratio}"
            )
        self.data_losses = []
        self.training_losses = []
        self.validation_losses = []
        self.best_model = None
        self.optimizer = torch.optim.AdamW(
            self.solution_u.parameters(),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = EKANScheduler(
            optimizer=self.optimizer,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr,
            iter_per_epoch=args.iter_per_epoch,
        )
        self.clear_logger = lambda: [
            h.close()
            for h in self.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]

    def Train(self, trainloader, validloader=None, testloader=None):
        """
        Training loop — pure MSE data loss on consecutive cycle pairs.
        Loss per batch: 0.5 * MSE(u1, y1) + 0.5 * MSE(u2, y2)
        """
        self.comp_tracker.start_training()
        best_val_mse = float("inf")
        best_epoch = 0
        early_stop_counter = 0
        print("Starting training — Data-Driven EKAN for Battery SOH Estimation")
        print(
            f"Early Stopping: Patience={self.args.early_stop}, Min Delta Ratio={self.args.early_stop_min_delta_ratio}"
        )
        for e in range(1, self.args.epochs + 1):
            self.solution_u.train()
            self.comp_tracker.start_epoch()
            meter_data = AverageMeter()
            for i, (x1, x2, y1, y2) in enumerate(trainloader, 1):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                u1 = self.solution_u(x1)
                u2 = self.solution_u(x2)
                data_loss = 0.5 * F.mse_loss(
                    u1, y1, reduction="mean"
                ) + 0.5 * F.mse_loss(u2, y2, reduction="mean")
                total_loss = data_loss
                if self.args.kan_regularization_weight > 0:
                    reg_loss = self.solution_u.regularization_loss(
                        self.args.kan_regularize_activation,
                        self.args.kan_regularize_entropy,
                    )
                    total_loss = (
                        total_loss + self.args.kan_regularization_weight * reg_loss
                    )
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.args.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.solution_u.parameters(), self.args.gradient_clip_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                meter_data.update(data_loss.item(), x1.size(0))
                if i % self.args.log_frequency == 0 and self.logger:
                    current_lr = self.scheduler.get_lr()
                    self.logger.info(
                        f"[Train] epoch:{e:4d}, iter:{i:4d}, data_loss:{data_loss.item():.8f}, lr:{current_lr:.6f}"
                    )
            epoch_mse = meter_data.avg
            self.data_losses.append(epoch_mse)
            self.training_losses.append(epoch_mse)
            current_epoch_time = self.comp_tracker.end_epoch()
            avg_epoch_time = self.comp_tracker.get_average_epoch_time()
            estimated_total_time = self.comp_tracker.get_total_training_time_estimate(
                self.args.epochs
            )
            current_training_time = self.comp_tracker.get_training_time()
            if self.logger:
                self.logger.info(
                    f"[Train] epoch:{e:4d}, data_loss:{epoch_mse:.8f}, lr:{self.scheduler.get_lr():.6f}"
                )
                self.logger.info(
                    f"[Timing] epoch_time:{current_epoch_time:.2f}s, avg_epoch_time:{avg_epoch_time:.2f}s, training_time:{current_training_time:.1f}s, estimated_total:{estimated_total_time:.1f}s"
                )
            if e % self.args.validation_frequency == 0 and validloader is not None:
                valid_mse, valid_rel_l2 = self.Valid(validloader)
                if self.logger:
                    self.logger.info(
                        f"[Valid] epoch:{e:4d}, MSE:{valid_mse:.8f}, Rel_L2:{valid_rel_l2:.8f}"
                    )
                    if len(self.data_losses) > 0 and len(self.validation_losses) > 0:
                        train_data_loss = self.data_losses[-1]
                        val_mse_last = self.validation_losses[-1]
                        loss_diff = abs(val_mse_last - train_data_loss)
                        loss_ratio = (
                            val_mse_last / train_data_loss
                            if train_data_loss > 1e-08
                            else float("inf")
                        )
                        fit_status, fit_details = ekan_detect_overfitting_underfitting(
                            self.data_losses, self.validation_losses
                        )
                        self.logger.info(
                            f"[Loss_Compare] epoch:{e}, Train_Loss:{train_data_loss:.7f}, Val_MSE:{val_mse_last:.7f}, Diff:{loss_diff:.7f}, Ratio:{loss_ratio:.3f}"
                        )
                        self.logger.info(
                            f"[Fit_Analysis] epoch:{e}, Status:{fit_status}"
                        )
                        if fit_status == "OVERFITTING":
                            self.logger.warning(
                                f"[OVERFITTING_DETECTED] Val MSE ({fit_details['val_loss']:.7f}) > Train Loss ({fit_details['train_loss']:.7f}) | Gap Ratio: {fit_details['loss_gap_ratio']:.3f}"
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
                if len(self.validation_losses) > 0:
                    self.validation_losses.append(self.validation_losses[-1])
                else:
                    self.validation_losses.append(self.data_losses[-1])
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
                        f"[NEW_BEST] New best validation MSE: {best_val_mse:.8f} at epoch {e} (improvement threshold: {min_delta:.8f})"
                    )
                if testloader is not None:
                    true_label, pred_label = self.Test(testloader)
                    MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true_label, pred_label)
                    test_rel_l2_error = calculate_relative_l2_error(
                        pred_label, true_label
                    )
                    if self.logger:
                        self.logger.info(
                            f"[Test] MSE:{MSE:.8f}, MAE:{MAE:.8f}, MAPE:{MAPE:.4f}%, RMSE:{RMSE:.8f}, R2:{R2:.8f}, Rel_L2:{test_rel_l2_error:.8f}"
                        )
                    final_fit_status, final_fit_details = (
                        ekan_detect_overfitting_underfitting(
                            self.data_losses, self.validation_losses
                        )
                    )
                    memory_stats = self.comp_tracker.get_memory_stats()
                    self.best_model = {
                        "solution_u": copy.deepcopy(self.solution_u.state_dict()),
                        "epoch": e,
                        "mse": MSE,
                        "mae": MAE,
                        "mape": MAPE,
                        "rmse": RMSE,
                        "r2": R2,
                        "relative_l2_error": test_rel_l2_error,
                        "data_losses": self.data_losses.copy(),
                        "training_losses": self.training_losses.copy(),
                        "validation_losses": self.validation_losses.copy(),
                        "fit_status": final_fit_status,
                        "fit_details": final_fit_details,
                        "epoch_times": self.comp_tracker.epoch_times.copy(),
                        "average_epoch_time": self.comp_tracker.get_average_epoch_time(),
                        "total_training_time": self.comp_tracker.get_training_time(),
                        "memory_stats": memory_stats,
                    }
                    if self.args.save_folder:
                        np.save(
                            os.path.join(self.args.save_folder, "true_label.npy"),
                            true_label,
                        )
                        np.save(
                            os.path.join(self.args.save_folder, "pred_label.npy"),
                            pred_label,
                        )
                        np.save(
                            os.path.join(self.args.save_folder, "data_losses.npy"),
                            np.array(self.data_losses),
                        )
                        np.save(
                            os.path.join(self.args.save_folder, "training_losses.npy"),
                            np.array(self.training_losses),
                        )
                        np.save(
                            os.path.join(
                                self.args.save_folder, "validation_losses.npy"
                            ),
                            np.array(self.validation_losses),
                        )
                        np.save(
                            os.path.join(self.args.save_folder, "epoch_times.npy"),
                            np.array(self.comp_tracker.epoch_times),
                        )
                        np.save(
                            os.path.join(self.args.save_folder, "cpu_memory_usage.npy"),
                            np.array(self.comp_tracker.cpu_memory_usage),
                        )
                        if self.comp_tracker.gpu_memory_usage:
                            np.save(
                                os.path.join(
                                    self.args.save_folder, "gpu_memory_allocated.npy"
                                ),
                                np.array(
                                    [
                                        m["allocated"]
                                        for m in self.comp_tracker.gpu_memory_usage
                                    ]
                                ),
                            )
                            np.save(
                                os.path.join(
                                    self.args.save_folder,
                                    "gpu_memory_max_allocated.npy",
                                ),
                                np.array(
                                    [
                                        m["max_allocated"]
                                        for m in self.comp_tracker.gpu_memory_usage
                                    ]
                                ),
                            )
                            np.save(
                                os.path.join(
                                    self.args.save_folder, "gpu_memory_reserved.npy"
                                ),
                                np.array(
                                    [
                                        m["reserved"]
                                        for m in self.comp_tracker.gpu_memory_usage
                                    ]
                                ),
                            )
                        torch.save(
                            self.best_model,
                            os.path.join(self.args.save_folder, "model.pth"),
                        )
            else:
                early_stop_counter += 1
                if self.logger:
                    self.logger.info(
                        f"[NO_IMPROVEMENT] No significant improvement (delta < {min_delta:.8f}). Early stop counter: {early_stop_counter}/{self.args.early_stop}"
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
            if self.logger:
                self.logger.info(
                    f"[RESTORE_BEST] Best solution_u weights restored from epoch {best_epoch}."
                )
        if testloader is not None:
            print(
                "\n[Inference Benchmark] Measuring prediction speed (solution_u only)"
            )
            self.solution_u.eval()
            sample_batch = next(iter(testloader))
            x_sample = sample_batch[0].to(self.device)
            with torch.no_grad():
                for _ in range(5):
                    _ = self.solution_u(x_sample)
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize()
            NUM_RUNS = 50
            times = []
            with torch.no_grad():
                for _ in range(NUM_RUNS):
                    t0 = time.perf_counter()
                    _ = self.solution_u(x_sample)
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
            avg_time_sec = sum(times) / len(times)
            n_samples = x_sample.shape[0]
            ms_per_1000 = avg_time_sec / n_samples * 1000000
            print(f"→ Batch size used            : {n_samples}")
            print(f"→ Average forward pass time  : {avg_time_sec * 1000:.3f} ms")
            print(
                f"→ Inference speed            : {ms_per_1000:.3f} ms per 1000 samples"
            )
            if self.best_model is not None:
                self.best_model["inference_time_ms_per_1000"] = ms_per_1000
                self.best_model["inference_batch_size_measured"] = n_samples
                self.best_model["inference_num_timing_runs"] = NUM_RUNS
                self.best_model["inference_avg_pass_ms"] = avg_time_sec * 1000
        final_training_time = self.comp_tracker.get_training_time()
        final_avg_epoch_time = self.comp_tracker.get_average_epoch_time()
        final_memory_stats = self.comp_tracker.get_memory_stats()
        if self.best_model and self.logger:
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"TRAINING COMPLETED — Best Model at Epoch {best_epoch}")
            self.logger.info(f"{'=' * 80}")
            self.logger.info(f"MSE:               {self.best_model['mse']:.8f}")
            self.logger.info(f"MAE:               {self.best_model['mae']:.8f}")
            self.logger.info(f"MAPE:              {self.best_model['mape']:.4f}%")
            self.logger.info(f"RMSE:              {self.best_model['rmse']:.8f}")
            self.logger.info(f"R2:                {self.best_model['r2']:.8f}")
            self.logger.info(
                f"Relative L2 Error: {self.best_model['relative_l2_error']:.8f}"
            )
            self.logger.info(f"Final Model Status: {self.best_model['fit_status']}")
            self.logger.info("\nCOMPUTATIONAL PERFORMANCE:")
            self.logger.info(
                f"Total Training Time: {final_training_time:.1f}s ({final_training_time / 60:.1f}min)"
            )
            self.logger.info(f"Average Epoch Time: {final_avg_epoch_time:.2f}s")
            self.logger.info(
                f"Total Epochs Trained: {len(self.comp_tracker.epoch_times)}"
            )
            self.logger.info("\nMEMORY USAGE:")
            self.logger.info(
                f"CPU Memory — Avg: {final_memory_stats['cpu_memory_avg']:.2f}GB, Peak: {final_memory_stats['cpu_memory_max']:.2f}GB"
            )
            if "gpu_memory_avg_allocated" in final_memory_stats:
                self.logger.info(
                    f"GPU Memory — Avg Allocated: {final_memory_stats['gpu_memory_avg_allocated']:.2f}GB, Peak Allocated: {final_memory_stats['gpu_memory_max_allocated']:.2f}GB"
                )
            if self.best_model.get("inference_time_ms_per_1000") is not None:
                self.logger.info("\nINFERENCE PERFORMANCE (solution_u only):")
                self.logger.info(
                    f"  Inference speed   : {self.best_model['inference_time_ms_per_1000']:.3f} ms per 1000 samples"
                )
                self.logger.info(
                    f"  Avg forward pass  : {self.best_model['inference_avg_pass_ms']:.3f} ms"
                )
            self.logger.info(f"{'=' * 80}\n")
        self.clear_logger()
        print("\nTraining completed — Data-Driven EKAN for Battery SOH.")

    def Valid(self, validloader):
        """Validation: compute data MSE (predictions vs. true SOH labels)."""
        self.solution_u.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for x1, x2, y1, y2 in validloader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)
                u = self.solution_u(x1)
                all_pred.append(u.cpu().numpy())
                all_true.append(y1.cpu().numpy())
        pred_label = np.concatenate(all_pred).flatten()
        true_label = np.concatenate(all_true).flatten()
        valid_mse = np.mean((pred_label - true_label) ** 2)
        rel_l2 = calculate_relative_l2_error(pred_label, true_label)
        self.validation_losses.append(valid_mse)
        return (valid_mse, rel_l2)

    def Test(self, testloader):
        """Test: return true and predicted SOH values."""
        self.solution_u.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for x1, x2, y1, y2 in testloader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)
                u = self.solution_u(x1)
                all_pred.append(u.cpu().numpy())
                all_true.append(y1.cpu().numpy())
        pred_label = np.concatenate(all_pred).flatten()
        true_label = np.concatenate(all_true).flatten()
        return (true_label, pred_label)


class BCPINNTrainer(BCPINN):
    """Training and evaluation for BCPINN; preserves the source loss protocol."""

    def __init__(self, args, seed=None, quiet=False):
        super().__init__(args, seed=seed)
        self.comp_tracker = ComputationalTracker()
        if args.save_folder and (not os.path.exists(args.save_folder)):
            os.makedirs(args.save_folder)
        log_path = (
            os.path.join(args.save_folder, "log.txt") if args.save_folder else None
        )
        self.logger = get_logger(log_path)
        if quiet:
            for h in self.logger.handlers[:]:
                if isinstance(h, logging.StreamHandler) and (
                    not isinstance(h, logging.FileHandler)
                ):
                    self.logger.removeHandler(h)
        self.logger.info(
            f"BC-PINN | beta(physics weight)={args.beta}, lambda_mono={args.lambda_mono} ({args.mono_reduction}), MC_T={args.mc_samples}, dropout={args.dropout}"
        )
        self.param_counts = {
            "params_solution_u": count_params(self.solution_u),
            "params_dynamical_F": count_params(self.dynamical_F),
            "params_calibrator": count_params(self.calibrator),
        }
        self.param_counts["params_inference_time"] = self.param_counts[
            "params_solution_u"
        ]
        self.param_counts["params_total_trainable"] = sum(
            (v for k, v in self.param_counts.items() if k != "params_inference_time")
        )
        self.logger.info(
            f"[PARAMS] solution_u={self.param_counts['params_solution_u']}, dynamical_F={self.param_counts['params_dynamical_F']}, calibrator={self.param_counts['params_calibrator']}, TOTAL(train-time)={self.param_counts['params_total_trainable']}, inference-time(solution_u only)={self.param_counts['params_inference_time']}"
        )
        self.data_losses, self.pde_losses, self.mono_losses = ([], [], [])
        self.validation_losses, self.training_losses = ([], [])
        self.best_model = None
        self.optimizer_main = torch.optim.AdamW(
            self.solution_u.parameters(),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay,
        )
        self.optimizer_aux = torch.optim.AdamW(
            list(self.dynamical_F.parameters()) + list(self.calibrator.parameters()),
            lr=args.lr_F,
            weight_decay=args.weight_decay,
        )
        self.scheduler = BCScheduler(
            self.optimizer_main,
            args.warmup_epochs,
            args.warmup_lr,
            args.epochs,
            args.lr,
            args.final_lr,
            iter_per_epoch=args.iter_per_epoch,
        )

    def clear_logger(self):
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()

    def Train(self, trainloader, validloader, testloader):
        self.comp_tracker.start_training()
        best_val_mse = float("inf")
        best_epoch = 0
        early_stop_counter = 0
        self.logger.info("Starting BC-PINN training (fixed-beta loss, MC Dropout).")
        for e in range(1, self.args.epochs + 1):
            self.solution_u.train()
            self.dynamical_F.train()
            self.calibrator.train()
            self.comp_tracker.start_epoch()
            meter_data, meter_pde, meter_mono = (
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
            )
            for x1, x2, y1, y2 in trainloader:
                x1, x2, y1, y2 = (t.to(self.device) for t in (x1, x2, y1, y2))
                u1, R1 = self.compute_calibrated_residual(x1)
                u2, R2 = self.compute_calibrated_residual(x2)
                zero1, zero2 = (torch.zeros_like(R1), torch.zeros_like(R2))
                data_loss = 0.5 * F.mse_loss(u1, y1) + 0.5 * F.mse_loss(u2, y2)
                physics_loss = 0.5 * F.mse_loss(R1, zero1) + 0.5 * F.mse_loss(R2, zero2)
                mono_loss = self.compute_monotonicity_loss(u1, u2, y1, y2)
                total_loss = (
                    data_loss
                    + self.args.beta * physics_loss
                    + self.args.lambda_mono * mono_loss
                )
                self.optimizer_main.zero_grad()
                self.optimizer_aux.zero_grad()
                total_loss.backward()
                if self.args.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.parameters(), self.args.gradient_clip_norm
                    )
                self.optimizer_main.step()
                self.optimizer_aux.step()
                self.scheduler.step()
                n = x1.size(0)
                meter_data.update(data_loss.item(), n)
                meter_pde.update(physics_loss.item(), n)
                meter_mono.update(mono_loss.item(), n)
            self.data_losses.append(meter_data.avg)
            self.pde_losses.append(meter_pde.avg)
            self.mono_losses.append(meter_mono.avg)
            self.training_losses.append(
                meter_data.avg
                + self.args.beta * meter_pde.avg
                + self.args.lambda_mono * meter_mono.avg
            )
            self.comp_tracker.end_epoch()
            self.logger.info(
                f"[Train] epoch:{e:4d}, data_loss:{meter_data.avg:.8f}, physics_loss:{meter_pde.avg:.8f}, mono_loss:{meter_mono.avg:.8f}, training_loss(total):{self.training_losses[-1]:.8f}, lr:{self.scheduler.get_lr():.6f}"
            )
            valid_mse, valid_rel_l2 = self.Valid(validloader)
            self.logger.info(
                f"[Valid] epoch:{e:4d}, validation_loss(MSE):{valid_mse:.8f}, Rel_L2:{valid_rel_l2:.8f}"
            )
            fit_status, fit_details = bc_detect_overfitting_underfitting(
                self.data_losses, self.validation_losses
            )
            self.logger.info(
                f"[FIT_STATUS] epoch:{e:4d}, status:{fit_status}, train_loss:{fit_details['train_loss']:.8f}, val_loss:{fit_details['val_loss']:.8f}, gap_ratio:{fit_details['loss_gap_ratio']:.4f}"
            )
            min_delta = (
                self.args.early_stop_min_delta_ratio * best_val_mse
                if best_val_mse < float("inf")
                else 1e-06
            )
            if valid_mse < best_val_mse - min_delta:
                best_val_mse = valid_mse
                best_epoch = e
                early_stop_counter = 0
                true_label, pred_label, pred_var = self.Test(testloader)
                MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true_label, pred_label)
                rel_l2 = calculate_relative_l2_error(pred_label, true_label)
                sigma_raw = np.sqrt(np.maximum(pred_var, 1e-12))
                s = sigma_scale_for_target_coverage(
                    true_label, pred_label, sigma_raw, s_max=3.0
                )
                sigma_cal = sigma_raw * s
                z = 1.96
                lower, upper = (pred_label - z * sigma_cal, pred_label + z * sigma_cal)
                coverage, mean_interval_width = coverage_and_width(
                    true_label, lower, upper
                )
                fit_status, _ = bc_detect_overfitting_underfitting(
                    self.data_losses, self.validation_losses
                )
                self.best_model = {
                    "solution_u": copy.deepcopy(self.solution_u.state_dict()),
                    "dynamical_F": copy.deepcopy(self.dynamical_F.state_dict()),
                    "calibrator": copy.deepcopy(self.calibrator.state_dict()),
                    "epoch": e,
                    "mse": MSE,
                    "mae": MAE,
                    "mape": MAPE,
                    "rmse": RMSE,
                    "r2": R2,
                    "relative_l2_error": rel_l2,
                    "coverage_rate": coverage,
                    "mean_interval_width": mean_interval_width,
                    "interval_scale": float(s),
                    "interval_calibration_split": "test (source protocol)",
                    "fit_status": fit_status,
                    "final_training_loss": self.training_losses[-1],
                    "final_data_loss": self.data_losses[-1],
                    "final_validation_loss": self.validation_losses[-1],
                }
                np.save(
                    os.path.join(self.args.save_folder, "true_label.npy"), true_label
                )
                np.save(
                    os.path.join(self.args.save_folder, "pred_label.npy"), pred_label
                )
                np.save(
                    os.path.join(self.args.save_folder, "pred_variance.npy"), pred_var
                )
                np.save(
                    os.path.join(self.args.save_folder, "interval_lower.npy"), lower
                )
                np.save(
                    os.path.join(self.args.save_folder, "interval_upper.npy"), upper
                )
                self.logger.info(
                    f"[NEW_BEST] epoch:{e}, MSE:{MSE:.8f}, MAPE:{MAPE:.4f}%, R2:{R2:.6f}, Coverage:{coverage:.4f}, MIW:{mean_interval_width:.6f}"
                )
            else:
                early_stop_counter += 1
            if self.args.early_stop and early_stop_counter >= self.args.early_stop:
                self.logger.info(f"[EARLY_STOP] epoch:{e}, best_epoch:{best_epoch}")
                break
        if self.best_model:
            self.solution_u.load_state_dict(self.best_model["solution_u"])
            self.dynamical_F.load_state_dict(self.best_model["dynamical_F"])
            self.calibrator.load_state_dict(self.best_model["calibrator"])
            self.logger.info(
                f"[RESTORE_BEST] Restored weights from epoch {best_epoch}."
            )
            self.best_model["num_epochs_trained"] = len(self.comp_tracker.epoch_times)
            self.best_model["total_training_time_s"] = (
                self.comp_tracker.get_training_time()
            )
            self.best_model["avg_epoch_time_s"] = (
                self.comp_tracker.get_average_epoch_time()
            )
            self.best_model.update(self.comp_tracker.get_memory_stats())
            self.best_model.update(self.benchmark_inference(testloader))
            self.best_model.update(self.param_counts)
            self.logger.info(
                f"[COMPUTE] train_time={self.best_model['total_training_time_s']:.1f}s, avg_epoch={self.best_model['avg_epoch_time_s']:.2f}s, deterministic_inference={self.best_model.get('inference_time_ms_per_1000', float('nan')):.3f} ms/1000, mc_dropout_inference(T={self.args.mc_samples})={self.best_model.get('mc_inference_time_ms_per_1000', float('nan')):.3f} ms/1000, params(solution_u/dynamical_F/calibrator/total)={self.param_counts['params_solution_u']}/{self.param_counts['params_dynamical_F']}/{self.param_counts['params_calibrator']}/{self.param_counts['params_total_trainable']}"
            )
        self.clear_logger()
        return self.best_model

    def benchmark_inference(self, testloader, num_warmup=5, num_runs=50):
        """
        Two inference-latency numbers, both needed to defend a fair
        comparison against PI-EKAN:

          1) 'deterministic' -- a single forward pass through solution_u
             with dropout OFF (directly comparable to a point-estimate
             model like PI-EKAN's forward pass).
          2) 'mc' -- the FULL cost of actually deploying BC-PINN as the
             paper intends: args.mc_samples stochastic forward passes to
             obtain a predictive mean/variance. This is the number that
             matters for a real inference-cost comparison, since BC-PINN's
             uncertainty quantification is not free.
        """
        try:
            x1, _, _, _ = next(iter(testloader))
        except StopIteration:
            return {}
        x1 = x1.to(self.device)
        self.solution_u.eval()
        self.solution_u.set_mc_dropout_mode(False)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.solution_u(x1)
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()
            det_times = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = self.solution_u(x1)
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize()
                det_times.append(time.perf_counter() - t0)
        avg_det = float(np.mean(det_times))
        n = x1.shape[0]
        with torch.no_grad():
            for _ in range(max(1, num_warmup // 2)):
                _ = self.predict_with_uncertainty(x1)
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()
            mc_times = []
            for _ in range(max(1, num_runs // 5)):
                t0 = time.perf_counter()
                _ = self.predict_with_uncertainty(x1)
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize()
                mc_times.append(time.perf_counter() - t0)
        avg_mc = float(np.mean(mc_times))
        return {
            "inference_batch_size_measured": n,
            "inference_num_timing_runs": num_runs,
            "inference_avg_pass_ms": avg_det * 1000.0,
            "inference_time_ms_per_1000": avg_det / n * 1000000,
            "mc_inference_mc_samples": self.args.mc_samples,
            "mc_inference_avg_pass_ms": avg_mc * 1000.0,
            "mc_inference_time_ms_per_1000": avg_mc / n * 1000000,
        }

    def Valid(self, validloader):
        self.solution_u.eval()
        all_pred, all_true = ([], [])
        with torch.no_grad():
            for x1, x2, y1, y2 in validloader:
                u = self.solution_u(x1.to(self.device))
                all_pred.append(u.cpu().numpy())
                all_true.append(y1.numpy())
        pred = np.concatenate(all_pred).flatten()
        true = np.concatenate(all_true).flatten()
        mse = float(np.mean((pred - true) ** 2))
        self.validation_losses.append(mse)
        return (mse, calculate_relative_l2_error(pred, true))

    def Test(self, testloader):
        """Returns (true, predictive_mean, predictive_variance) using MC Dropout."""
        all_true, all_mu, all_var = ([], [], [])
        for x1, x2, y1, y2 in testloader:
            mu, var = self.predict_with_uncertainty(x1)
            all_mu.append(mu.cpu().numpy())
            all_var.append(var.cpu().numpy())
            all_true.append(y1.numpy())
        return (
            np.concatenate(all_true).flatten(),
            np.clip(np.concatenate(all_mu).flatten(), 0.0, 1.0),
            np.concatenate(all_var).flatten(),
        )


BC_PINN_SEARCH_SPACE = {
    "beta": [0.01, 0.05, 0.1, 0.5, 1.0],
    "lambda_mono": [0.05, 0.1, 0.2, 0.5],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [0.001, 0.0005, 0.0001],
    "encoder_hidden_dim": [32, 60, 100],
    "F_hidden_dim": [32, 50],
}
FIXED_BC_PINN_HP = dict(
    beta=0.05,
    lambda_mono=0.5,
    dropout=0.1,
    lr=0.001,
    encoder_hidden_dim=60,
    F_hidden_dim=50,
)


def random_search_bc_pinn(base_args, data, n_trials, search_epochs, seed_base=1000):
    """
    Identical protocol to the sequence baselines' random_search(): n_trials
    random draws, each trained for a short FIXED number of epochs (early
    stopping disabled during search so the budget is strictly comparable),
    best trial selected by validation MSE.
    """
    rng = random.Random(20260904)
    best_hp, best_val = (None, float("inf"))
    print(
        f"\n--- BC-PINN hyperparameter search ({n_trials} trials, {search_epochs} epochs each) ---"
    )
    for trial in range(n_trials):
        set_seed(seed_base + trial)
        hp = {k: rng.choice(v) for k, v in BC_PINN_SEARCH_SPACE.items()}
        trial_args = copy.deepcopy(base_args)
        for k, v in hp.items():
            setattr(trial_args, k, v)
        trial_args.epochs = search_epochs
        trial_args.early_stop = 0
        trial_args.save_folder = os.path.join(
            base_args.save_folder, "_search", f"trial_{trial}"
        )
        model = BCPINNTrainer(trial_args, seed=seed_base + trial, quiet=True)
        for _ in range(search_epochs):
            model.solution_u.train()
            model.dynamical_F.train()
            model.calibrator.train()
            for x1, x2, y1, y2 in data["train"]:
                x1, x2, y1, y2 = (t.to(model.device) for t in (x1, x2, y1, y2))
                u1, R1 = model.compute_calibrated_residual(x1)
                u2, R2 = model.compute_calibrated_residual(x2)
                loss = (
                    0.5 * F.mse_loss(u1, y1)
                    + 0.5 * F.mse_loss(u2, y2)
                    + trial_args.beta
                    * (
                        0.5 * F.mse_loss(R1, torch.zeros_like(R1))
                        + 0.5 * F.mse_loss(R2, torch.zeros_like(R2))
                    )
                    + trial_args.lambda_mono
                    * model.compute_monotonicity_loss(u1, u2, y1, y2)
                )
                model.optimizer_main.zero_grad()
                model.optimizer_aux.zero_grad()
                loss.backward()
                if trial_args.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), trial_args.gradient_clip_norm
                    )
                model.optimizer_main.step()
                model.optimizer_aux.step()
                model.scheduler.step()
        val_mse, _ = model.Valid(data["valid"])
        model.clear_logger()
        print(f"  Trial {trial + 1}/{n_trials}: hp={hp} -> val MSE={val_mse:.6f}")
        if val_mse < best_val:
            best_val, best_hp = (val_mse, hp)
        del model
        gc.collect()
    print(f"  -> Best BC-PINN hyperparameters: {best_hp} (val MSE={best_val:.6f})")
    return (best_hp, best_val)


def write_spec(save_dir, args, param_counts):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "bc_pinn_spec.txt"), "w") as f:
        f.write("=== BC-PINN Specification ===\n")
        f.write(
            f"Bayesian SOH estimator: encoder_layers={args.encoder_layers_num}, encoder_hidden={args.encoder_hidden_dim}, encoder_out={args.encoder_output_dim}, predictor_hidden={args.predictor_hidden_dim}, dropout={args.dropout}\n"
        )
        f.write(
            f"Governing-eq. approximator G_theta: layers={args.F_layers_num}, hidden={args.F_hidden_dim}\n"
        )
        f.write(
            f"Physics calibrator s_psi: layers={args.calibrator_layers_num}, hidden={args.calibrator_hidden_dim}, dropout={args.calibrator_dropout}\n"
        )
        f.write("Activation: Sin (solution_u/G_theta), Tanh (calibrator)\n")
        f.write("Loss: L_total = L_data + beta * L_Cphysics + lambda_mono * L_mono\n")
        f.write(
            f"      beta={args.beta} (FIXED, not adaptive), lambda_mono={args.lambda_mono}, mono reduction='{args.mono_reduction}' (L_mono = ReLU((u2-u1)*(y1-y2)) — identical form to PI-EKAN/PINN)\n"
        )
        if args.search:
            f.write(
                f"Hyperparameter tuning: {args.n_trials} random-search trials x {args.search_epochs} epochs (IDENTICAL to the sequence baselines)\n"
            )
        else:
            f.write(
                f"Hyperparameter tuning: FIXED (not searched this run). Values were selected via an earlier {args.n_trials}-trial random search and are held fixed across datasets, consistent with PI-EKAN's own single-dataset sensitivity-analysis protocol.\n"
            )
        f.write(f"MC Dropout samples at inference: {args.mc_samples}\n")
        f.write(
            f"Optimizer: AdamW (weight_decay={args.weight_decay}); solution_u: warmup+cosine LR ({args.warmup_lr}->{args.lr}->{args.final_lr}); G_theta+calibrator: constant LR ({args.lr_F})\n"
        )
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Max epochs: {args.epochs}\n")
        f.write(
            f"Early stopping: patience={args.early_stop}, min_delta_ratio={args.early_stop_min_delta_ratio}\n"
        )
        f.write(f"Trainable parameters: {param_counts}\n")
        f.write(
            "DEVIATION FROM PAPER (1): G_theta is trained jointly on real data (no PET-simulation pretraining/freezing stage available) -- see file docstring.\n"
        )
        f.write(
            "DEVIATION FROM PAPER (2): a monotonicity term (identical to PI-EKAN's) was ADDED to the published data+physics loss, so BC-PINN receives the same loss ingredients as PI-EKAN rather than being handicapped. Set --lambda_mono 0.0 for the paper-faithful ablation.\n"
        )
        f.write(
            "\nPer-run RMSE/MAE/MAPE/R2/coverage, total training time, average epoch time, peak CPU/GPU memory, deterministic (dropout-off) inference latency, and full MC-Dropout predictive-pass latency (T={}) are written to each experiment's results.txt so BC-PINN can be defended against PI-EKAN on accuracy AND on compute cost (note: MC-Dropout inference is inherently ~T x more expensive than a single-pass point estimate -- report both numbers, not just the deterministic one, to avoid an unfair inference-cost comparison).\n".format(
                args.mc_samples
            )
        )


class BaselineTrainer:

    def __init__(
        self,
        model,
        model_name,
        train_loader,
        valid_loader,
        test_loader,
        epochs,
        batch_size,
        warmup_epochs,
        warmup_lr,
        lr,
        final_lr,
        weight_decay,
        early_stop,
        early_stop_min_delta_ratio,
        gradient_clip_norm,
        save_dir,
        log_frequency=50,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.early_stop = early_stop
        self.early_stop_min_delta_ratio = early_stop_min_delta_ratio
        self.gradient_clip_norm = gradient_clip_norm
        self.save_dir = save_dir
        self.log_frequency = log_frequency
        os.makedirs(save_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(save_dir, "log.txt"), name=model_name)
        self.loss_func = nn.MSELoss()
        self.comp_tracker = ComputationalTracker(model_name)
        self.train_losses, self.val_losses = ([], [])
        self.param_count = count_params(self.model)
        self.logger.info(
            f"[PARAMS] model={model_name}, trainable_parameters={self.param_count}"
        )
        print(f"  [PARAMS] {model_name}: {self.param_count} trainable parameters")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=warmup_lr, weight_decay=weight_decay
        )
        self.scheduler = SEQScheduler(
            self.optimizer,
            warmup_epochs,
            warmup_lr,
            epochs,
            lr,
            final_lr,
            iter_per_epoch=len(train_loader),
        )

    def clear_logger(self):
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        self.comp_tracker.start_epoch()
        self.comp_tracker.reset_memory_tracking()
        meter = AverageMeter()
        for xb, yb in self.train_loader:
            xb, yb = (xb.to(self.device), yb.to(self.device))
            y_pred = self.model(xb)
            loss = self.loss_func(y_pred, yb)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
            self.optimizer.step()
            meter.update(loss.item(), xb.size(0))
        self.scheduler.step()
        train_loss = meter.avg
        self.train_losses.append(train_loss)
        self.comp_tracker.end_epoch()
        self.logger.info(
            f"[Train] epoch:{epoch:04d}, loss:{train_loss:.8f}, lr:{self.scheduler.get_lr():.6f}"
        )
        return train_loss

    def valid(self, epoch):
        self.model.eval()
        meter = AverageMeter()
        with torch.no_grad():
            for xb, yb in self.valid_loader:
                xb, yb = (xb.to(self.device), yb.to(self.device))
                loss = F.mse_loss(self.model(xb), yb)
                meter.update(loss.item(), xb.size(0))
        val_loss = meter.avg
        self.val_losses.append(val_loss)
        self.logger.info(f"[Valid] epoch:{epoch:04d}, loss:{val_loss:.8f}")
        return val_loss

    def test(self):
        self.model.eval()
        true_list, pred_list = ([], [])
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb = xb.to(self.device)
                pred_list.append(self.model(xb).cpu().numpy())
                true_list.append(yb.numpy())
        return (np.concatenate(true_list, axis=0), np.concatenate(pred_list, axis=0))

    def benchmark_inference(self, num_warmup=5, num_runs=50):
        """Measures pure forward-pass latency of the trained model on one
        test batch (same protocol as the EKAN/IAW-PI-EKAN inference
        benchmark: warm-up passes, then timed runs, reported as ms per
        1000 samples so it is directly comparable across models/datasets)."""
        self.model.eval()
        try:
            xb, _ = next(iter(self.test_loader))
        except StopIteration:
            return {}
        xb = xb.to(self.device)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(xb)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = self.model(xb)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        avg_time_sec = float(np.mean(times))
        n_samples = xb.shape[0]
        return {
            "inference_batch_size_measured": n_samples,
            "inference_num_timing_runs": num_runs,
            "inference_avg_pass_ms": avg_time_sec * 1000.0,
            "inference_time_ms_per_1000": avg_time_sec / n_samples * 1000000,
        }

    def train(self):
        self.comp_tracker.start_training()
        min_val_loss = float("inf")
        early_stop_counter = 0
        best_results = None
        best_epoch = 0
        best_state = None
        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)
            val_loss = self.valid(epoch)
            min_delta = (
                self.early_stop_min_delta_ratio * min_val_loss
                if min_val_loss < float("inf")
                else 1e-06
            )
            if val_loss < min_val_loss - min_delta:
                best_state = copy.deepcopy(self.model.state_dict())
                min_val_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                true_label, pred_label = self.test()
                MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true_label, pred_label)
                rel_l2 = calculate_relative_l2_error(pred_label, true_label)
                best_results = {
                    "MSE": MSE,
                    "MAE": MAE,
                    "MAPE": MAPE,
                    "RMSE": RMSE,
                    "R2": R2,
                    "Relative_L2_Error": rel_l2,
                    "epoch": epoch,
                }
                self.logger.info(
                    f"[NEW_BEST] epoch:{epoch}, MSE:{MSE:.8f}, MAPE:{MAPE:.4f}%, R2:{R2:.6f}"
                )
            else:
                early_stop_counter += 1
            if self.early_stop and early_stop_counter >= self.early_stop:
                self.logger.info(f"[EARLY_STOP] epoch:{epoch}, best_epoch:{best_epoch}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(
                {"state_dict": best_state, "epoch": best_epoch},
                os.path.join(self.save_dir, "model.pth"),
            )
        true_label, pred_label = self.test()
        np.save(os.path.join(self.save_dir, "true_label.npy"), true_label)
        np.save(os.path.join(self.save_dir, "pred_label.npy"), pred_label)
        fit_status, fit_details = seq_detect_overfitting_underfitting(
            self.train_losses, self.val_losses
        )
        self.logger.info(f"[FIT_STATUS] {fit_status} | {fit_details}")
        self.clear_logger()
        if best_results is None:
            true_label, pred_label = self.test()
            MAE, MAPE, MSE, RMSE, R2 = eval_metrics(true_label, pred_label)
            best_results = {
                "MSE": MSE,
                "MAE": MAE,
                "MAPE": MAPE,
                "RMSE": RMSE,
                "R2": R2,
                "Relative_L2_Error": calculate_relative_l2_error(
                    pred_label, true_label
                ),
                "epoch": self.epochs,
            }
        best_results["fit_status"] = fit_status
        best_results["trainable_params"] = count_params(self.model)
        best_results["total_training_time_s"] = self.comp_tracker.get_training_time()
        best_results["avg_epoch_time_s"] = self.comp_tracker.get_average_epoch_time()
        best_results["num_epochs_trained"] = len(self.comp_tracker.epoch_times)
        best_results.update(self.comp_tracker.get_memory_stats())
        best_results.update(self.benchmark_inference())
        return best_results


SEARCH_SPACES = {
    "LSTM": {
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "predictor_hidden": [32, 64],
        "lr": [0.001, 0.0005, 0.0001],
    },
    "Transformer": {
        "hidden_dim": [32, 64, 128],
        "nhead": [4],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "predictor_hidden": [32, 64],
        "lr": [0.001, 0.0005, 0.0001],
    },
    "LSTM-KAN": {
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "kan_grid_size": [3, 5],
        "kan_spline_order": [3],
        "lr": [0.001, 0.0005, 0.0001],
    },
    "RNN-KAN": {
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "kan_grid_size": [3, 5],
        "kan_spline_order": [3],
        "lr": [0.001, 0.0005, 0.0001],
    },
}
FIXED_HYPERPARAMETERS = {
    "LSTM": dict(
        hidden_dim=128, num_layers=2, dropout=0.2, predictor_hidden=64, lr=0.001
    ),
    "Transformer": dict(
        hidden_dim=32,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        predictor_hidden=64,
        lr=0.0005,
    ),
    "LSTM-KAN": dict(
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        kan_grid_size=5,
        kan_spline_order=3,
        lr=0.001,
    ),
    "RNN-KAN": dict(
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        kan_grid_size=5,
        kan_spline_order=3,
        lr=0.0005,
    ),
}


def sample_hparams(model_name, rng):
    space = SEARCH_SPACES[model_name]
    hp = {k: rng.choice(v) for k, v in space.items()}
    return hp


def random_search(
    model_name, input_dim, data, n_trials, search_epochs, batch_size, seed_base=1000
):
    """
    Identical protocol for all four models: n_trials random draws from the
    model's own (comparably-sized) search space, each trained for a short,
    FIXED number of epochs (search_epochs) with early stopping disabled
    during search (to keep the budget strictly comparable), best trial
    chosen by validation MSE.
    """
    rng = random.Random(hash(model_name) % 2**31)
    best_hp, best_val = (None, float("inf"))
    print(
        f"\n--- Hyperparameter search: {model_name} ({n_trials} trials, {search_epochs} epochs each) ---"
    )
    for trial in range(n_trials):
        set_seed(seed_base + trial)
        hp = sample_hparams(model_name, rng)
        model = build_baseline(model_name, input_dim, hp).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-05)
        for _ in range(search_epochs):
            model.train()
            for xb, yb in data["train"]:
                xb, yb = (xb.to(device), yb.to(device))
                opt.zero_grad()
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in data["valid"]:
                xb, yb = (xb.to(device), yb.to(device))
                val_losses.append(F.mse_loss(model(xb), yb).item())
        val_mse = float(np.mean(val_losses))
        print(f"  Trial {trial + 1}/{n_trials}: hp={hp} -> val MSE={val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            best_hp = hp
    print(f"  -> Best {model_name} hyperparameters: {best_hp} (val MSE={best_val:.6f})")
    return (best_hp, best_val)


def write_baseline_spec(
    save_dir,
    model_name,
    hp,
    input_dim,
    window_size,
    args,
    n_trials,
    param_count,
    was_searched=True,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "baseline_spec_table3.txt")
    with open(path, "w") as f:
        f.write(f"=== Baseline Specification (Table 3) — {model_name} ===\n")
        f.write(
            f"Sequence length (window W):   {window_size} (FIXED for all sequence models and all datasets; not searched per dataset)\n"
        )
        f.write(f"Input dim per time step:      {input_dim}\n")
        f.write(f"Hyperparameters:              {hp}\n")
        f.write(f"Dropout:                      {hp.get('dropout')}\n")
        f.write(
            f"Activation:                   {('GELU (Transformer FFN)' if model_name == 'Transformer' else 'LeakyReLU/SiLU (KAN base) as applicable')}\n"
        )
        f.write(
            f"Optimizer:                    AdamW (weight_decay={args.weight_decay})\n"
        )
        f.write(f"Gradient clip norm:           {args.gradient_clip_norm}\n")
        f.write(
            f"LR schedule:                  warmup_lr={args.warmup_lr} -> lr={hp.get('lr')} -> final_lr={args.final_lr}, warmup_epochs={args.warmup_epochs}\n"
        )
        f.write(f"Batch size:                   {args.batch_size}\n")
        f.write(f"Max epochs:                   {args.epochs}\n")
        f.write(
            f"Early stopping:               patience={args.early_stop}, min_delta_ratio={args.early_stop_min_delta_ratio}\n"
        )
        f.write(f"Trainable parameters:         {param_count}\n")
        if was_searched:
            f.write(
                f"Hyperparameter tuning budget: {n_trials} random-search trials x {args.search_epochs} epochs (IDENTICAL for all baselines)\n"
            )
        else:
            f.write(
                f"Hyperparameter tuning:        FIXED (not searched this run). Values were selected via an earlier {n_trials}-trial random search, with manual overrides documented in FIXED_HYPERPARAMETERS in this file. Held fixed across datasets for consistency with the PI-EKAN sensitivity-analysis protocol (single-dataset search, reused across chemistries).\n"
            )
        f.write(
            "\nPer-run RMSE/MAE/MAPE/R2, total training time, average epoch time, peak CPU/GPU memory, and inference latency (ms per 1000 samples) are written to each experiment's results.txt (see run folder) so every baseline can be defended against PI-EKAN on accuracy AND compute cost.\n"
        )
    return path


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _save_run(trainer, args, data, results, model_name):
    """Keep complete histories separate from validation-selected checkpoint data."""
    import json
    from pathlib import Path
    from .tracking import benchmark_forward, close_logger

    output = Path(args.save_folder)
    tracker = trainer.comp_tracker
    tracker.stop_training()
    summary = {
        k: v
        for k, v in results.items()
        if isinstance(v, (str, int, float, bool, np.generic)) or v is None
    }
    summary.update(
        model=model_name,
        total_training_time_s=tracker.get_training_time(),
        avg_epoch_time_s=tracker.get_average_epoch_time(),
        num_epochs_trained=len(tracker.epoch_times),
        **tracker.get_memory_stats(),
    )
    histories = {}
    for name, value in vars(trainer).items():
        if isinstance(value, list) and (
            name.endswith("_losses") or name.endswith("_history")
        ):
            histories[name] = value
            np.save(output / f"{name}.npy", np.asarray(value))
    for name in ("epoch_times", "cpu_memory_usage", "gpu_memory_usage"):
        histories[name] = getattr(tracker, name)
    (output / "history.json").write_text(
        json.dumps(histories, indent=2, default=_json_value), encoding="utf-8"
    )
    np.save(output / "epoch_times.npy", np.asarray(tracker.epoch_times))
    np.save(output / "cpu_memory_usage.npy", np.asarray(tracker.cpu_memory_usage))
    if model_name == "PI-EKAN":
        x = next(iter(data["test"]))[0].to(trainer.device)
        summary.update(benchmark_forward(trainer.solution_u, x))
    prediction_model = (
        trainer.model if isinstance(trainer, BaselineTrainer) else trainer
    )
    torch.save(prediction_model.state_dict(), output / "model_state.pth")
    summary["total_trainable_params"] = count_params(prediction_model)
    if not isinstance(trainer, BaselineTrainer):
        if not trainer.best_model:
            raise RuntimeError(
                "No finite improving validation checkpoint was produced."
            )
        torch.save(trainer.best_model, output / "model.pth")
    (output / "results.json").write_text(
        json.dumps(summary, indent=2, default=_json_value), encoding="utf-8"
    )
    (output / "results.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )
    (output / "computational_summary.txt").write_text(
        "\n".join(
            f"{k}: {v}"
            for k, v in summary.items()
            if any(term in k for term in ("time", "memory", "epoch", "params"))
        )
        + "\n",
        encoding="utf-8",
    )
    close_logger(trainer.logger)
    print(f"Completed {model_name}: results saved to {output}")
    return summary


def run_experiment(
    model_name, args, dataset_cfg, experiment_id, small_sample=None, hp=None
):
    """Execute one seeded run using the shared data pipeline."""
    import json
    from pathlib import Path
    from dataloader.dataloader import load_dataset, save_data_manifest
    from .tracking import close_logger

    args = copy.deepcopy(args)
    set_seed(42 + experiment_id)
    is_sequence = model_name in FIXED_HYPERPARAMETERS
    data = load_dataset(
        args,
        dataset_cfg,
        small_sample,
        mode="sequence" if is_sequence else "pairs",
        window_size=getattr(args, "window_size", 20),
    )
    tag = (
        model_name
        if small_sample is None
        else f"{model_name}_Small_Sample_{small_sample}"
    )
    output = (
        Path(args.results_root)
        / tag
        / dataset_cfg["results_tag"]
        / f"Experiment_{experiment_id}"
    )
    output.mkdir(parents=True, exist_ok=True)
    args.save_folder = str(output)
    args.log_dir = "log.txt"
    args.input_dim = data["input_dim"]
    args.F_input_dim = 2 * (args.input_dim - 1) + 3
    args.iter_per_epoch = data["iter_per_epoch"]
    args.device = device
    save_data_manifest(data, dataset_cfg, output)
    config = dict(
        vars(args),
        model=model_name,
        seed=42 + experiment_id,
        small_sample=small_sample,
        hyperparameters=hp,
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, default=_json_value), encoding="utf-8"
    )
    trainer = None
    try:
        if is_sequence:
            model = build_baseline(model_name, args.input_dim, hp)
            trainer = BaselineTrainer(
                model,
                model_name,
                data["train"],
                data["valid"],
                data["test"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                warmup_epochs=args.warmup_epochs,
                warmup_lr=args.warmup_lr,
                lr=hp["lr"],
                final_lr=args.final_lr,
                weight_decay=args.weight_decay,
                early_stop=args.early_stop,
                early_stop_min_delta_ratio=args.early_stop_min_delta_ratio,
                gradient_clip_norm=args.gradient_clip_norm,
                save_dir=str(output),
            )
            write_baseline_spec(
                str(output),
                model_name,
                hp,
                args.input_dim,
                args.window_size,
                args,
                args.n_trials,
                count_params(model),
                was_searched=args.search,
            )
            results = trainer.train()
        else:
            classes = {
                "PI-EKAN": PIEKANTrainer,
                "PINN": PINNTrainer,
                "EKAN": EKANTrainer,
                "BC-PINN": BCPINNTrainer,
            }
            trainer = classes[model_name](args, seed=42 + experiment_id)
            trainer.comp_tracker.start_experiment()
            if model_name == "BC-PINN":
                write_spec(str(output), args, trainer.param_counts)
            trainer.Train(data["train"], data["valid"], data["test"])
            results = trainer.best_model or {}
        return _save_run(trainer, args, data, results, model_name)
    finally:
        if trainer is not None:
            close_logger(trainer.logger)
        for loader in (data["train"], data["valid"], data["test"]):
            if hasattr(loader, "_cached_gpu_data"):
                del loader._cached_gpu_data
        gc.collect()


def run_cli(model_name, defaults, argv=None):
    """Shared CLI; defaults remain in the five readable entry-point files."""
    import argparse
    import json
    from pathlib import Path
    from dataloader.dataloader import DATASET_CONFIGS, get_dataset_config, load_dataset

    parser = argparse.ArgumentParser(
        description=f"{model_name} battery SOH experiments"
    )
    parser.add_argument("--dataset", default="1", help="Dataset name, ID (1–8), or all")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--dataset-dir", help="Exact CSV directory for a single dataset"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--small-samples",
        type=int,
        nargs="+",
        help="Training cell counts, e.g. 1 2 3 4",
    )
    parser.add_argument(
        "--normalization", choices=["min-max", "z-score", "none"], default=None
    )
    parser.add_argument(
        "--config", help="JSON object overriding model/training defaults"
    )
    parser.add_argument("--threads", type=int, help="Optional CPU thread limit")
    if model_name == "baseline":
        parser.add_argument(
            "--models",
            nargs="+",
            choices=list(FIXED_HYPERPARAMETERS),
            default=list(FIXED_HYPERPARAMETERS),
        )
    exposed = {
        "epochs",
        "batch_size",
        "warmup_epochs",
        "results_root",
        "lr",
        "early_stop",
        "n_trials",
        "search_epochs",
        "window_size",
    }
    for key in sorted(exposed & vars(defaults).keys()):
        value = getattr(defaults, key)
        parser.add_argument(
            "--" + key.replace("_", "-"),
            "--" + key,
            dest=key,
            type=type(value),
            default=None,
        )
    if hasattr(defaults, "search"):
        parser.add_argument("--search", action="store_true", default=None)
    options = parser.parse_args(argv)
    if options.config:
        overrides = json.loads(Path(options.config).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            parser.error("--config must contain a JSON object.")
        unknown = set(overrides) - vars(defaults).keys()
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        for key, value in overrides.items():
            setattr(defaults, key, value)
    for key in exposed | {"search"}:
        if getattr(options, key, None) is not None:
            setattr(defaults, key, getattr(options, key))
    defaults.normalization_method = options.normalization or getattr(
        defaults, "normalization_method", "min-max"
    )
    defaults.device = device
    if options.threads is not None:
        if options.threads < 1:
            parser.error("--threads must be positive.")
        torch.set_num_threads(options.threads)
    if options.runs < 1 or defaults.epochs < 1 or defaults.batch_size < 1:
        parser.error("Runs, epochs, and batch size must be positive.")
    if (
        getattr(defaults, "outlier_removal", True) is not True
        or getattr(defaults, "outlier_sigma", 3.0) != 3.0
    ):
        parser.error(
            "This source-preserving pipeline uses feature-only 3-sigma filtering."
        )
    if defaults.warmup_epochs < 0:
        parser.error("Warmup epochs cannot be negative.")
    if hasattr(defaults, "gamma") and (defaults.gamma <= 0 or defaults.sigma_init <= 0):
        parser.error("gamma and sigma_init must be positive.")
    if getattr(defaults, "validation_frequency", 1) != 1:
        parser.error("This checkpoint protocol requires validation_frequency=1.")
    if options.dataset.lower() == "all" and options.dataset_dir:
        parser.error("--dataset-dir requires one selected dataset.")
    datasets = (
        list(DATASET_CONFIGS) if options.dataset.lower() == "all" else [options.dataset]
    )
    models = options.models if model_name == "baseline" else [model_name]
    sizes = options.small_samples or [None]
    all_results = []
    print(f'Device: {device} | models: {", ".join(models)} | runs: {options.runs}')
    for dataset in datasets:
        cfg = get_dataset_config(
            dataset,
            options.data_root or getattr(defaults, "data_root", "data"),
            options.dataset_dir,
        )
        for name in models:
            args = copy.deepcopy(defaults)
            hp = (
                dict(FIXED_HYPERPARAMETERS[name])
                if name in FIXED_HYPERPARAMETERS
                else None
            )
            if getattr(args, "search", False):
                set_seed(42)
                if args.n_trials < 1 or args.search_epochs < 1:
                    parser.error("Search trials and epochs must be positive.")
                probe = load_dataset(
                    args,
                    cfg,
                    mode="sequence" if hp else "pairs",
                    window_size=getattr(args, "window_size", 20),
                )
                if hp:
                    hp, score = random_search(
                        name,
                        probe["input_dim"],
                        probe,
                        args.n_trials,
                        args.search_epochs,
                        args.batch_size,
                    )
                else:
                    args.input_dim = probe["input_dim"]
                    args.F_input_dim = 2 * (args.input_dim - 1) + 3
                    args.iter_per_epoch = probe["iter_per_epoch"]
                    args.save_folder = str(
                        Path(args.results_root) / name / cfg["results_tag"]
                    )
                    hp_search, score = random_search_bc_pinn(
                        args, probe, args.n_trials, args.search_epochs
                    )
                    for key, value in hp_search.items():
                        setattr(args, key, value)
                if not np.isfinite(score):
                    raise RuntimeError(
                        "Hyperparameter search produced no finite validation score."
                    )
            for size in sizes:
                run_results = []
                for experiment_id in range(1, options.runs + 1):
                    print(
                        f'\n{name} | {cfg["name"]} | cells={size or "all"} | run {experiment_id}/{options.runs}'
                    )
                    result = run_experiment(name, args, cfg, experiment_id, size, hp)
                    run_results.append(result)
                    all_results.append(
                        dict(
                            result,
                            dataset=cfg["name"],
                            small_sample=size,
                            run=experiment_id,
                        )
                    )
                for metric in (
                    "mse",
                    "rmse",
                    "mae",
                    "mape",
                    "r2",
                    "MSE",
                    "RMSE",
                    "MAE",
                    "MAPE",
                    "R2",
                ):
                    values = [r[metric] for r in run_results if metric in r]
                    if values:
                        print(
                            f"{metric}: mean={np.mean(values):.8f}, std={np.std(values):.8f}"
                        )
    output = Path(defaults.results_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{model_name}_summary.json").write_text(
        json.dumps(all_results, indent=2, default=_json_value), encoding="utf-8"
    )
