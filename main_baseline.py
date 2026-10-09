"""Configuration and command-line runner."""

import argparse


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--warmup_lr", type=float, default=1e-05)
    p.add_argument("--final_lr", type=float, default=1e-06)
    p.add_argument("--weight_decay", type=float, default=1e-05)
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--early_stop_min_delta_ratio", type=float, default=0.001)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument(
        "--n_trials",
        type=int,
        default=8,
        help="Hyperparameter-search trial budget (SAME for all 4 baselines).",
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
        help="Re-run the random hyperparameter search instead of using the FIXED_HYPERPARAMETERS values already selected for LSTM/Transformer/LSTM-KAN/RNN-KAN. Off by default.",
    )
    p.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Fixed sliding-window length (cycles) used by ALL sequence models on ALL datasets. No per-dataset search is performed.",
    )
    p.add_argument("--results_root", type=str, default="./results_sequence_baselines")
    return p.parse_args([])


def main(argv=None):
    from util.training import run_cli

    run_cli("baseline", get_config(), argv)


if __name__ == "__main__":
    main()
