"""Single-cell transfer settings.

Place this folder beside the PI-EKAN project's existing dataloader folder.
Edit DATA_ROOT, PRETRAINED_ROOT and RESULTS_ROOT for your machine.
Checkpoints are PRETRAINED_ROOT/<source chemistry>/model.pth.

The original feature windows and cell pools are retained. In particular,
NCM and LCO use their supplied "NCA 15 %- 63% Window Selected" directories.
Five runs and 100 fixed epochs are the effective defaults of the supplied code.
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(
    os.environ.get("PI_EKAN_DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser()
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained_models"
RESULTS_ROOT = PROJECT_ROOT / "results" / "single_cell_adaptation"

DATASET_CONFIGS = {
    "NCM": {
        "path": str(DATA_ROOT / "sources chemistry"),
        "nominal_capacity": 2.0,
        "is_already_normalized": False,
        "cell_pool": [
            "2C_battery-1",
            "2C_battery-2",
            "2C_battery-3",
            "2C_battery-4",
            "2C_battery-5",
            "2C_battery-6",
            "2C_battery-7",
            "2C_battery-8",
        ],
    },
    "NCA": {
        "path": str(DATA_ROOT / "sources chemistry"),
        "nominal_capacity": 3.5,
        "is_already_normalized": False,
        "cell_pool": [
            "CY25-05_1-#1",
            "CY25-05_1-#2",
            "CY25-05_1-#3",
            "CY25-05_1-#4",
            "CY25-05_1-#5",
            "CY25-05_1-#6",
            "CY25-05_1-#7",
            "CY25-05_1-#10",
            "CY25-05_1-#11",
            "CY25-05_1-#12",
            "CY25-05_1-#14",
            "CY25-05_1-#13",
            "CY25-05_1-#15",
        ],
    },
    "LFP": {
        "path": str(DATA_ROOT / "sources chemistry"),
        "nominal_capacity": 1.1,
        "is_already_normalized": False,
        "cell_pool": [
            "SNL_18650_LFP_25C_0-100_0.5-3C_a",
            "SNL_18650_LFP_15C_0-100_0.5-2C_b",
            "SNL_18650_LFP_25C_0-100_0.5-0.5C_a",
            "SNL_18650_LFP_25C_0-100_0.5-2C_a",
            "SNL_18650_LFP_25C_0-100_0.5-2C_b",
            "SNL_18650_LFP_25C_0-100_0.5-3C_b",
            "SNL_18650_LFP_25C_0-100_0.5-3C_c",
            "SNL_18650_LFP_25C_0-100_0.5-3C_d",
            "SNL_18650_LFP_35C_0-100_0.5-1C_a",
            "SNL_18650_LFP_35C_0-100_0.5-1C_b",
            "SNL_18650_LFP_35C_0-100_0.5-1C_d",
            "SNL_18650_LFP_35C_0-100_0.5-2C_b",
            "SNL_18650_LFP_25C_0-100_0.5-1C_a",
            "SNL_18650_LFP_25C_0-100_0.5-1C_b",
            "SNL_18650_LFP_35C_0-100_0.5-1C_c",
        ],
    },
    "NA-ion": {
        "path": str(DATA_ROOT / "sources chemistry"),
        "nominal_capacity": 1.0,
        "is_already_normalized": False,
        "cell_pool": [
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
            "NA-ion_270040-8-1-20",
            "NA-ion_270040-6-7-25",
            "NA-ion_270040-4-1-48",
            "NA-ion_270040-8-8-13",
            "NA-ion_270040-6-1-31",
            "NA-ion_270040-6-3-29",
            "NA-ion_270040-6-4-28",
        ],
    },
    "LCO": {
        "path": str(DATA_ROOT / "sources chemistry"),
        "nominal_capacity": 1.1,
        "is_already_normalized": False,
        "cell_pool": [
            "CALCE_CS2_35",
            "CALCE_CS2_36",
            "CALCE_CS2_37",
            "CALCE_CS2_38",
            "CALCE_CS2_33",
            "CALCE_CS2_34",
        ],
    },
}

TRANSFER_MAP = {
    "NCM": ["LFP", "NCA", "LCO"],
    "LFP": ["NCM", "NCA", "LCO"],
    "NCA": ["NCM", "LFP", "LCO"],
    "LCO": ["NCM", "NCA", "LFP"],
    "NA-ion": ["NCM", "NCA", "LFP", "LCO"],
}
TRANSFER_PAIRS = tuple(
    (source, target) for source, targets in TRANSFER_MAP.items() for target in targets
)


class TransferArgs:

    def __init__(self):
        self.results_root = str(RESULTS_ROOT)
        self.pretrained_root = str(PRETRAINED_ROOT)
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
        self.adaptation_epochs = 100
        self.adaptation_lr = 0.01
        self.batch_size = 1024
        self.gamma = 100.0
        self.sigma_init = 1.0
        self.sigma_lr = 0.001
        self.sigma_weight_decay = 0.0
        self.freeze_adaptive_weights = True
        self.normalization_method = "min-max"
        self.gradient_clip_norm = 1.0
        self.weight_decay = 0.01
        self.kan_regularization_weight = 0.0
        self.kan_regularize_activation = 1.0
        self.kan_regularize_entropy = 1.0
        self.lora_rank = 32
        self.lora_alpha = 64
        self.adapter_variant = "hybrid"
        self.n_shot = 1
        self.n_runs = 5
        self.adaptation_seed = 368144
        self.save_folder = None
        self.log_dir = "transfer_log.txt"
        self.iter_per_epoch = 1
        self.batch = None


def get_args():
    return TransferArgs()
