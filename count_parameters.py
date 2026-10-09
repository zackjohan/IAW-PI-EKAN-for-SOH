"""Count trainable, total, and prediction-network parameters for all models."""

import argparse
import json

from main_pi_ekan import get_config as pi_config
from main_pinn import get_config as pinn_config
from main_ekan import get_config as ekan_config
from main_bc_pinn import get_config as bc_config
from model import PIEKAN, PINN, EKAN, BCPINN, build_baseline
from util.training import FIXED_HYPERPARAMETERS
from util.util import set_seed


def parameter_report(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prediction_network = getattr(model, "solution_u", model)
    report = {
        "total": total,
        "trainable": trainable,
        "prediction_network": sum(p.numel() for p in prediction_network.parameters()),
    }
    for name in ("solution_u", "dynamical_F", "calibrator"):
        if hasattr(model, name):
            report[name] = sum(p.numel() for p in getattr(model, name).parameters())
    report["adaptive_scalars"] = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("log_sigma_squared_")
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print a JSON table")
    args = parser.parse_args(argv)
    set_seed(42)
    report = {}
    for name, cls, factory in [
        ("PI-EKAN", PIEKAN, pi_config),
        ("PINN", PINN, pinn_config),
        ("EKAN", EKAN, ekan_config),
        ("BC-PINN", BCPINN, bc_config),
    ]:
        report[name] = parameter_report(cls(factory()))
    for name, hp in FIXED_HYPERPARAMETERS.items():
        report[name] = parameter_report(build_baseline(name, 8, hp))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f'{"Model":<15} {"Total":>12} {"Trainable":>12} {"Prediction":>12} {"Adaptive":>10}'
        )
        for name, row in report.items():
            print(
                f'{name:<15} {row["total"]:>12,} {row["trainable"]:>12,} '
                f'{row["prediction_network"]:>12,} {row["adaptive_scalars"]:>10,}'
            )
        print("Default configurations, input_dim=8. Buffers are excluded.")
        print(
            "BC-PINN prediction count excludes its calibration network; MC passes reuse weights."
        )


if __name__ == "__main__":
    main()
