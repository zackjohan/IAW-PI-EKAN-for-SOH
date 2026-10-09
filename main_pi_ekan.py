"""Configuration and command-line runner."""


class Config:

    def __init__(self):
        self.data_root = "data"
        self.results_root = "results"
        self.data = "MultiDataset"
        # Network dimensions
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
        # Training
        self.epochs = 500
        self.batch_size = 256
        self.early_stop = 20
        self.early_stop_min_delta_ratio = 0.001
        self.weight_decay = 1e-05
        # Bounded adaptive weighting
        self.gamma = 100.0
        self.sigma_init = 1.0
        self.sigma_lr = 0.0004
        self.sigma_weight_decay = 0.0
        self.scheduler = "cosine"
        # Learning-rate schedule
        self.warmup_epochs = 10
        self.warmup_lr = 0.005
        self.lr = 0.01
        self.final_lr = 0.0002
        self.lr_F = 0.001
        # Shared preprocessing
        self.normalization_method = "min-max"
        self.outlier_removal = True
        self.outlier_sigma = 3.0
        # Optimization safeguards and regularization
        self.gradient_clip_norm = 1.0
        self.kan_regularization_weight = 0.001
        self.kan_regularize_activation = 1.0
        self.kan_regularize_entropy = 1.0
        # Reporting
        self.log_frequency = 50
        self.validation_frequency = 1
        # Runtime fields
        self.batch = None
        self.save_folder = None
        self.log_dir = "logs"
        self.iter_per_epoch = 1

    def update_from_main(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_config():
    return Config()


def main(argv=None):
    from util.training import run_cli

    run_cli("PI-EKAN", get_config(), argv)


if __name__ == "__main__":
    main()
