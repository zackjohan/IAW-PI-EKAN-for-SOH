"""Paths, dataset definitions, training defaults, and ablation configurations."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("PI_EKAN_DATA_ROOT", str(BASE_DIR / "data"))
).expanduser()
OUT_ROOT = str(BASE_DIR / "results")
SEEDS = 10
FEATURE_NAMES = None


def dataset_path(relative):
    return str(DATA_ROOT / relative)


NCM = 1
NCA = 2
NCM_NCA = 3
LFP = 4
LCO = 5
NCM_LCO = 6
NA_ION = 7
STANFORD = 8
DATASET_NAMES = {
    NCM: "NCM",
    NCA: "NCA",
    NCM_NCA: "NCM_NCA",
    LFP: "LFP",
    LCO: "LCO",
    NCM_LCO: "NCM_LCO",
    NA_ION: "NA-ion",
    STANFORD: "Stanford",
}
ALL_DATASETS = [NCM, NCA, NCM_NCA, LFP, LCO, NCM_LCO, NA_ION, STANFORD]
DATASETS_4 = [NCM, LFP, STANFORD, LCO]
FAMILY_A_DATASETS = [LFP]
FAMILY_B_DATASETS = [LFP]
FAMILY_C_DATASETS = ALL_DATASETS.copy()
FAMILY_D_DATASETS = [NCM, LFP, STANFORD, NCM_LCO, LCO, NA_ION]
FAMILY_F_DATASETS = ALL_DATASETS.copy()
WINDOW_FOLDERS = {
    STANFORD: {
        "0-100": dataset_path("Stanford_NCA/0 %- 100% Window Selected"),
        "20-80": dataset_path("Stanford_NCA/20 %- 80% Window Selected"),
        "25-75": dataset_path("Stanford_NCA/25 %- 75% Window Selected"),
        "30-70": dataset_path("Stanford_NCA/30 %- 70% Window Selected"),
        "40-60": dataset_path("Stanford_NCA/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("Stanford_NCA/PS-BO Window Selected"),
    },
    NCM: {
        "0-100": dataset_path("NCM/0 %- 100% Window Selected"),
        "20-80": dataset_path("NCM/20 %- 80% Window Selected"),
        "25-75": dataset_path("NCM/25 %- 75% Window Selected"),
        "30-70": dataset_path("NCM/30 %- 70% Window Selected"),
        "40-60": dataset_path("NCM/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("NCM/PS-BO Window Selected"),
    },
    NCA: {
        "0-100": dataset_path("NCA/0 %- 100% Window Selected"),
        "20-80": dataset_path("NCA/20 %- 80% Window Selected"),
        "25-75": dataset_path("NCA/25 %- 75% Window Selected"),
        "30-70": dataset_path("NCA/30 %- 70% Window Selected"),
        "40-60": dataset_path("NCA/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("NCA/PS-BO Window Selected"),
    },
    NCM_NCA: {
        "0-100": dataset_path("NCM_NCA/0 %- 100% Window Selected"),
        "20-80": dataset_path("NCM_NCA/20 %- 80% Window Selected"),
        "25-75": dataset_path("NCM_NCA/25 %- 75% Window Selected"),
        "30-70": dataset_path("NCM_NCA/30 %- 70% Window Selected"),
        "40-60": dataset_path("NCM_NCA/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("NCM_NCA/PS-BO Window Selected"),
    },
    LFP: {
        "0-100": dataset_path("LFP/0 %- 100% Window Selected"),
        "20-80": dataset_path("LFP/20 %- 80% Window Selected"),
        "25-75": dataset_path("LFP/25 %- 75% Window Selected"),
        "30-70": dataset_path("LFP/30 %- 70% Window Selected"),
        "40-60": dataset_path("LFP/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("LFP/PS-BO Window Selected"),
    },
    LCO: {
        "0-100": dataset_path("LCO/0 %- 100% Window Selected"),
        "20-80": dataset_path("LCO/20 %- 80% Window Selected"),
        "25-75": dataset_path("LCO/25 %- 75% Window Selected"),
        "30-70": dataset_path("LCO/30 %- 70% Window Selected"),
        "40-60": dataset_path("LCO/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("LCO/PS-BO Window Selected"),
    },
    NCM_LCO: {
        "0-100": dataset_path("NCM_LCO/0 %- 100% Window Selected"),
        "20-80": dataset_path("NCM_LCO/20 %- 80% Window Selected"),
        "25-75": dataset_path("NCM_LCO/25 %- 75% Window Selected"),
        "30-70": dataset_path("NCM_LCO/30 %- 70% Window Selected"),
        "40-60": dataset_path("NCM_LCO/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("NCM_LCO/PS-BO Window Selected"),
    },
    NA_ION: {
        "0-100": dataset_path("NA_ion/0 %- 100% Window Selected"),
        "20-80": dataset_path("NA_ion/20 %- 80% Window Selected"),
        "25-75": dataset_path("NA_ion/25 %- 75% Window Selected"),
        "30-70": dataset_path("NA_ion/30 %- 70% Window Selected"),
        "40-60": dataset_path("NA_ion/40 %- 60% Window Selected"),
        "PS-BO": dataset_path("NA-ion/PS-BO Window Selected"),
    },
}
SKIP_WINDOW_LABELS = {"PS-BO"}
PSBO_FEATURE_PATHS = {
    key: folders["PS-BO"]
    for key, folders in WINDOW_FOLDERS.items()
    if "PS-BO" in folders
}
REFERENCE_FOLDERS = {
    NCM: {
        "consensus": dataset_path(
            "NCM/PSBO_Modules/03_Reference_Sensitivity/features/consensus_42-96"
        ),
        "maxupper": dataset_path(
            "NCM/PSBO_Modules/03_Reference_Sensitivity/features/maxupper_3C_battery-1_37-100"
        ),
        "medscore": dataset_path(
            "NCM/PSBO_Modules/03_Reference_Sensitivity/features/medscore_3C_battery-7_45-97"
        ),
        "minlower": dataset_path(
            "NCM/PSBO_Modules/03_Reference_Sensitivity/features/minlower_3C_battery-3_18-42"
        ),
    },
    LFP: {
        "consensus": dataset_path(
            "LFP/PSBO_Modules/03_Reference_Sensitivity/features/consensus_47-86"
        ),
        "maxupper": dataset_path(
            "LFP/PSBO_Modules/03_Reference_Sensitivity/features/maxupper_SNL_18650_LFP_35C_0-100_0.5-2C_b_52-92"
        ),
        "medscore": dataset_path(
            "LFP/PSBO_Modules/03_Reference_Sensitivity/features/medscore_SNL_18650_LFP_35C_0-100_0.5-1C_b_41-80"
        ),
        "minlower": dataset_path(
            "LFP/PSBO_Modules/03_Reference_Sensitivity/features/minlower_SNL_18650_LFP_25C_0-100_0.5-2C_b_18-74"
        ),
    },
    STANFORD: {
        "consensus": dataset_path(
            "Stanford_NCA/PSBO_Modules/03_Reference_Sensitivity/features/consensus_48-100"
        ),
        "maxupper": dataset_path(
            "Stanford_NCA/PSBO_Modules/03_Reference_Sensitivity/features/maxupper_Cell_060.csv_49-100"
        ),
        "medscore": dataset_path(
            "Stanford_NCA/PSBO_Modules/03_Reference_Sensitivity/features/medscore_Cell_070.csv_37-100"
        ),
        "minlower": dataset_path(
            "Stanford_NCA/PSBO_Modules/03_Reference_Sensitivity/features/minlower_Cell_069.csv_37-100"
        ),
    },
    LCO: {
        "consensus": dataset_path(
            "LCO/PSBO_Modules/03_Reference_Sensitivity/features/consensus_25-96"
        ),
        "maxupper": dataset_path(
            "LCO/PSBO_Modules/03_Reference_Sensitivity/features/maxupper_CALCE_CS2_35_34-100"
        ),
        "medscore": dataset_path(
            "LCO/PSBO_Modules/03_Reference_Sensitivity/features/medscore_CALCE_CS2_37_20-100"
        ),
        "minlower": dataset_path(
            "LCO/PSBO_Modules/03_Reference_Sensitivity/features/minlower_CALCE_CX2_36_9-92"
        ),
    },
}
DATASET_CONFIGS = {
    1: {
        "name": "NCM",
        "path": PSBO_FEATURE_PATHS[1],
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
        "path": PSBO_FEATURE_PATHS[2],
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
        "path": PSBO_FEATURE_PATHS[3],
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
        "path": PSBO_FEATURE_PATHS[4],
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
        "path": PSBO_FEATURE_PATHS[5],
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
        "path": PSBO_FEATURE_PATHS[6],
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
        "path": PSBO_FEATURE_PATHS[7],
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
        "path": PSBO_FEATURE_PATHS[8],
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


class Config:

    def __init__(self):
        self.data_root = "data"
        self.results_root = "results"
        self.data = "MultiDataset"
        self.input_dim = 8
        self.output_dim = 1
        self.kan_hidden_layers = [8, 20, 1]
        self.kan_grid_size = 5
        self.kan_spline_order = 3
        self.kan_scale_noise = 0.1
        self.kan_scale_base = 1.0
        self.kan_scale_spline = 1.0
        self.kan_grid_eps = 0.02
        self.kan_grid_range = [-1, 1]
        self.F_layers_num = 3
        self.F_hidden_dim = 10
        self.F_input_dim = 2 * (self.input_dim - 1) + 3
        self.F_output_dim = 1
        self.F_kan_grid_size = 5
        self.F_kan_spline_order = 3
        self.F_kan_scale_noise = 0.1
        self.F_kan_scale_base = 1.0
        self.F_kan_scale_spline = 1.0
        self.F_kan_grid_eps = 0.02
        self.F_kan_grid_range = [-1, 1]
        self.epochs = 500
        self.batch_size = 256
        self.early_stop = 20
        self.early_stop_min_delta_ratio = 0.001
        self.weight_decay = 1e-05
        self.gamma = 100.0
        self.sigma_init = 1.0
        self.sigma_lr = 0.0004
        self.sigma_weight_decay = 0.0
        self.scheduler = "cosine"
        self.warmup_epochs = 10
        self.warmup_lr = 0.005
        self.lr = 0.01
        self.final_lr = 0.0002
        self.lr_F = 0.001
        self.normalization_method = "min-max"
        self.outlier_removal = True
        self.outlier_sigma = 3.0
        self.gradient_clip_norm = 1.0
        self.kan_regularization_weight = 0.001
        self.kan_regularize_activation = 1.0
        self.kan_regularize_entropy = 1.0
        self.log_frequency = 50
        self.validation_frequency = 1
        self.batch = None
        self.save_folder = None
        self.log_dir = "logs"
        self.iter_per_epoch = 1

    def update_from_main(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


DEFAULT_CFG = dict(
    name="PI_EKAN",
    family="C",
    answers="",
    where="",
    panel="",
    use_features=True,
    use_cycle=True,
    keep_features=None,
    drop_features=None,
    pde=True,
    mono="label_aware",
    g_inputs="full",
    adaptive=True,
    bounded=True,
    fixed_lambdas=(1.0, 1.0, 1.0),
    gamma=None,
    basis="spline",
    freeze_g=False,
    path_override=None,
    datasets=None,
)


def make_cfg(**kw):
    cfg = dict(DEFAULT_CFG)
    unknown = set(kw) - set(DEFAULT_CFG)
    if unknown:
        raise KeyError(f"unknown config field(s): {sorted(unknown)}")
    cfg.update(kw)
    return cfg


def build_registry(feature_names):
    registry = {}
    A = [
        make_cfg(
            name="A1_cycle_only",
            family="A",
            answers="cycle-only baseline",
            where="MAIN",
            panel="12d",
            use_features=False,
            use_cycle=True,
            datasets=FAMILY_A_DATASETS,
        ),
        make_cfg(
            name="A2_features_only",
            family="A",
            answers="no-cycle-index ablation",
            where="MAIN",
            panel="12d",
            use_features=True,
            use_cycle=False,
            pde=False,
            datasets=FAMILY_A_DATASETS,
        ),
    ]
    for f in feature_names:
        A.append(
            make_cfg(
                name=f"A7_LOFO_{f}",
                family="A",
                answers="validates SHAP/KAN attribution by intervention",
                where="MAIN",
                panel="12i",
                drop_features=[f],
                datasets=FAMILY_A_DATASETS,
            )
        )
    registry["A"] = A
    registry["B"] = [
        make_cfg(
            name="B1b_G_no_ut",
            family="B",
            g_inputs="no_ut",
            answers="removes the trivial-residual path",
            where="MAIN",
            panel="12e",
            datasets=FAMILY_B_DATASETS,
        ),
        make_cfg(
            name="B1c_G_no_ux",
            family="B",
            g_inputs="no_ux",
            answers="do feature gradients matter?",
            where="MAIN",
            panel="12e",
            datasets=FAMILY_B_DATASETS,
        ),
        make_cfg(
            name="B1d_G_t_u",
            family="B",
            g_inputs="t_u",
            answers="is a state-and-time law sufficient?",
            where="MAIN",
            panel="12e",
            datasets=FAMILY_B_DATASETS,
        ),
        make_cfg(
            name="B1e_G_u_only",
            family="B",
            g_inputs="u_only",
            answers="pure autonomous decay",
            where="MAIN",
            panel="12e",
            datasets=FAMILY_B_DATASETS,
        ),
        make_cfg(
            name="B3_G_frozen",
            family="B",
            freeze_g=True,
            answers="is G learned or just a regulariser?",
            where="MAIN",
            panel="12e",
            datasets=FAMILY_B_DATASETS,
        ),
    ]
    C = [
        make_cfg(
            name="C1_pde_fixed",
            family="C",
            pde=True,
            mono="off",
            adaptive=False,
            fixed_lambdas=(1.0, 1.0, 0.0),
            answers="PDE residual alone",
            where="MAIN",
            panel="12a/b/c",
            datasets=FAMILY_C_DATASETS,
        ),
        make_cfg(
            name="C2_mono_fixed",
            family="C",
            pde=False,
            mono="label_aware",
            adaptive=False,
            fixed_lambdas=(1.0, 0.0, 1.0),
            answers="monotonicity alone",
            where="MAIN",
            panel="12a/b/c",
            datasets=FAMILY_C_DATASETS,
        ),
        make_cfg(
            name="C3_pde_mono_fixed",
            family="C",
            pde=True,
            mono="label_aware",
            adaptive=False,
            fixed_lambdas=(1.0, 1.0, 1.0),
            answers="both terms, fixed weights",
            where="MAIN",
            panel="12a/b/c",
            datasets=FAMILY_C_DATASETS,
        ),
        make_cfg(
            name="C4_pde_adaptive",
            family="C",
            pde=True,
            mono="off",
            adaptive=True,
            bounded=True,
            answers="PDE + adaptive weighting",
            where="MAIN",
            panel="12a/b/c",
            datasets=FAMILY_C_DATASETS,
        ),
        make_cfg(
            name="C5_mono_adaptive",
            family="C",
            pde=False,
            mono="label_aware",
            adaptive=True,
            bounded=True,
            answers="mono + adaptive weighting",
            where="MAIN",
            panel="12a/b/c",
            datasets=FAMILY_C_DATASETS,
        ),
        make_cfg(
            name="C7_unbounded",
            family="C",
            pde=True,
            mono="label_aware",
            adaptive=True,
            bounded=False,
            answers="isolates the bound gamma from Kendall weighting",
            where="MAIN",
            panel="12a/b/c",
            datasets=FAMILY_C_DATASETS,
        ),
    ]
    registry["C"] = C
    D = []
    for dkey, folders in WINDOW_FOLDERS.items():
        if dkey not in FAMILY_D_DATASETS:
            continue
        for label, path in folders.items():
            if label in SKIP_WINDOW_LABELS:
                continue
            D.append(
                make_cfg(
                    name=f"D_window_{label}",
                    family="D",
                    answers="fixed vs adaptive window",
                    where="SI",
                    panel="S_D1",
                    path_override=path,
                    datasets=[dkey],
                )
            )
    registry["D"] = D
    E = []
    for dkey, folders in REFERENCE_FOLDERS.items():
        for label, path in folders.items():
            E.append(
                make_cfg(
                    name=f"E_{label}",
                    family="E",
                    answers="reference-cell sensitivity",
                    where="MAIN",
                    panel="12g",
                    path_override=path,
                    datasets=[dkey],
                )
            )
    registry["E"] = E
    registry["F"] = [
        make_cfg(
            name="F1_spline",
            family="F",
            basis="spline",
            answers="B-spline (proposed)",
            where="MAIN",
            panel="12f",
            datasets=FAMILY_F_DATASETS,
        ),
        make_cfg(
            name="F2_fourier",
            family="F",
            basis="fourier",
            answers="global Fourier prior",
            where="MAIN",
            panel="12f",
            datasets=FAMILY_F_DATASETS,
        ),
        make_cfg(
            name="F4_wavelet",
            family="F",
            basis="wavelet",
            answers="wavelet: locality without splines",
            where="MAIN",
            panel="12f",
            datasets=FAMILY_F_DATASETS,
        ),
        make_cfg(
            name="F5_legendre",
            family="F",
            basis="legendre",
            answers="global polynomial basis",
            where="MAIN",
            panel="12f",
            datasets=FAMILY_F_DATASETS,
        ),
    ]
    return registry


FIG_OUT = os.path.join(OUT_ROOT, "figures")
REF_DATASETS = [DATASET_NAMES[k] for k in DATASETS_4]
DATASET_ORDER = [DATASET_NAMES[k] for k in ALL_DATASETS]
WINDOW_PANEL_DATASET = None
PSBO_WINDOW_BOUNDS = {}
RUNS_DIR = os.path.join(OUT_ROOT, "runs")
CSV_DIR = os.path.join(OUT_ROOT, "csv")
DIAG_DIR = os.path.join(OUT_ROOT, "diagnostics")
