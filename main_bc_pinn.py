"""Configuration and command-line runner."""

import argparse


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--warmup_lr", type=float, default=1e-05)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--final_lr", type=float, default=1e-06)
    p.add_argument("--lr_F", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-05)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--early_stop_min_delta_ratio", type=float, default=0.001)
    p.add_argument("--encoder_output_dim", type=int, default=32)
    p.add_argument("--encoder_layers_num", type=int, default=3)
    p.add_argument("--encoder_hidden_dim", type=int, default=60)
    p.add_argument("--predictor_hidden_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--F_layers_num", type=int, default=4)
    p.add_argument("--F_hidden_dim", type=int, default=50)
    p.add_argument("--F_output_dim", type=int, default=1)
    p.add_argument("--calibrator_layers_num", type=int, default=3)
    p.add_argument("--calibrator_hidden_dim", type=int, default=32)
    p.add_argument("--calibrator_dropout", type=float, default=0.1)
    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Fixed physics-loss weight (paper Eq. 15).",
    )
    p.add_argument(
        "--lambda_mono",
        type=float,
        default=0.2,
        help="Weight on the monotonicity term (project parity with PI-EKAN). Set 0.0 to recover the paper-faithful data+physics loss.",
    )
    p.add_argument(
        "--mono_reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="'mean' matches IAW-PI-EKAN; 'sum' matches PINN.py.",
    )
    p.add_argument(
        "--mc_samples",
        type=int,
        default=50,
        help="MC Dropout forward passes at inference.",
    )
    p.add_argument(
        "--n_trials",
        type=int,
        default=8,
        help="Random-search trial budget. Keep IDENTICAL to the sequence baselines.",
    )
    p.add_argument(
        "--search_epochs",
        type=int,
        default=20,
        help="Short training length used only during hyperparameter search.",
    )
    p.add_argument(
        "--search",
        action="store_true",
        default=False,
        help="Re-run the random hyperparameter search instead of using FIXED_BC_PINN_HP. Off by default.",
    )
    p.add_argument("--results_root", type=str, default="./results_bc_pinn")
    args = p.parse_args([])
    from util.training import FIXED_BC_PINN_HP

    for key, value in FIXED_BC_PINN_HP.items():
        setattr(args, key, value)
    args.input_dim = 8
    args.F_input_dim = 17
    args.iter_per_epoch = 1
    args.save_folder = None
    return args


def main(argv=None):
    from util.training import run_cli

    run_cli("BC-PINN", get_config(), argv)


if __name__ == "__main__":
    main()
