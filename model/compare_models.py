import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from util.util import set_seed


class PINNSin(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class PINNMLP(nn.Module):

    def __init__(
        self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2
    ):
        super().__init__()
        assert layers_num >= 2, "layers_num must be >= 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers += [nn.Linear(input_dim, hidden_dim), PINNSin()]
            elif i == layers_num - 1:
                layers += [nn.Linear(hidden_dim, output_dim)]
            else:
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    PINNSin(),
                    nn.Dropout(p=dropout),
                ]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class PINNPredictor(nn.Module):

    def __init__(self, input_dim=32, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, hidden_dim),
            PINNSin(),
            nn.Linear(hidden_dim, 1),
        )
        self.input_dim = input_dim

    def forward(self, x):
        return self.net(x)


class PINNSolution_u(nn.Module):

    def __init__(
        self,
        input_dim=8,
        encoder_output_dim=32,
        encoder_layers_num=3,
        encoder_hidden_dim=60,
        predictor_hidden_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = PINNMLP(
            input_dim=input_dim,
            output_dim=encoder_output_dim,
            layers_num=encoder_layers_num,
            hidden_dim=encoder_hidden_dim,
            dropout=dropout,
        )
        self.predictor = PINNPredictor(
            input_dim=encoder_output_dim, hidden_dim=predictor_hidden_dim
        )
        self._init_()

    def get_embedding(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)


class PINN(nn.Module):
    """PINN architecture and model-specific mathematics."""

    def __init__(self, args, seed=None):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.args = args
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.solution_u = PINNSolution_u(
            input_dim=args.input_dim,
            encoder_output_dim=args.encoder_output_dim,
            encoder_layers_num=args.encoder_layers_num,
            encoder_hidden_dim=args.encoder_hidden_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            dropout=args.dropout,
        ).to(self.device)
        self.dynamical_F = PINNMLP(
            input_dim=args.F_input_dim,
            output_dim=args.F_output_dim,
            layers_num=args.F_layers_num,
            hidden_dim=args.F_hidden_dim,
            dropout=args.dropout,
        ).to(self.device)
        self.relu = nn.ReLU()

    def compute_loss(self, data_loss, pde_loss, monotonicity_loss):
        """Compute total loss using constant weights."""
        return (
            self.args.alpha_data * data_loss
            + self.args.alpha_pde * pde_loss
            + self.args.alpha_mono * monotonicity_loss
        )

    def compute_pde_residual(self, xt):
        """
        Compute the integer-order PDE residual  f = ∂u/∂t − G(t, x, u, u_t, u_x).

        Gradients are computed w.r.t. the full input tensor xt then sliced,
        ensuring the autograd graph is correctly connected to the original
        leaf tensor (via detach + requires_grad_) and avoids stale graph
        accumulation across calls.

        Args:
            xt : [batch, input_dim] — feature columns followed by cycle_index (t)

        Returns:
            u  : [batch, 1] — SOH prediction
            f  : [batch, 1] — PDE residual
        """
        xt = xt.detach().requires_grad_(True)
        u = self.solution_u(xt)
        all_gradients = grad(
            outputs=u,
            inputs=xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = all_gradients[:, :-1]
        u_t = all_gradients[:, -1:]
        if u_t is None:
            u_t = torch.zeros_like(xt[:, -1:])
        if u_x is None:
            u_x = torch.zeros_like(xt[:, :-1])
        rhs_input = torch.cat([xt, u, u_x, u_t], dim=1)
        F_out = self.dynamical_F(rhs_input)
        f = u_t - F_out
        return (u, f)

    def forward(self, x):
        return self.solution_u(x)


class EKANSin(nn.Module):

    def __init__(self):
        super(EKANSin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class EKANKANLinear(nn.Module):

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
        base_activation=EKANSin,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(EKANKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (
                grid[:, k:-1] - grid[:, : -(k + 1)]
            ) * bases[:, :, :-1] + (grid[:, k + 1 :] - x) / (
                grid[:, k + 1 :] - grid[:, 1:-k]
            ) * bases[
                :, :, 1:
            ]
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class EKANEfficientKAN(nn.Module):

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=EKANSin,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(EKANEfficientKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                EKANKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            (
                layer.regularization_loss(regularize_activation, regularize_entropy)
                for layer in self.layers
            )
        )


class EKAN(nn.Module):
    """EKAN architecture and model-specific mathematics."""

    def __init__(self, args, seed=None):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.args = args
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.solution_u = EKANEfficientKAN(
            layers_hidden=args.kan_hidden_layers,
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            scale_noise=args.kan_scale_noise,
            scale_base=args.kan_scale_base,
            scale_spline=args.kan_scale_spline,
            base_activation=EKANSin,
            grid_eps=args.kan_grid_eps,
            grid_range=args.kan_grid_range,
        ).to(self.device)

    def forward(self, x):
        return self.solution_u(x)


class BCSin(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class BCMLP(nn.Module):
    """Generic fully-connected block with MC-Dropout-compatible Dropout
    layers, matching PINN.py's MLP design."""

    def __init__(self, input_dim, output_dim, layers_num=4, hidden_dim=50, dropout=0.2):
        super().__init__()
        assert layers_num >= 2
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers += [nn.Linear(input_dim, hidden_dim), BCSin()]
            elif i == layers_num - 1:
                layers += [nn.Linear(hidden_dim, output_dim)]
            else:
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    BCSin(),
                    nn.Dropout(p=dropout),
                ]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class BCBayesianPredictor(nn.Module):
    """Prediction head with an explicit Dropout layer used for MC sampling."""

    def __init__(self, input_dim=32, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_dim),
            BCSin(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class BCBayesianSolutionU(nn.Module):
    """The Bayesian SOH estimator u(t,x). Dropout is present in both the
    encoder and the predictor so MC-Dropout uncertainty reflects the whole
    network, as in the paper's Section 3.3(1)."""

    def __init__(
        self,
        input_dim=8,
        encoder_output_dim=32,
        encoder_layers_num=3,
        encoder_hidden_dim=60,
        predictor_hidden_dim=32,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = BCMLP(
            input_dim,
            encoder_output_dim,
            encoder_layers_num,
            encoder_hidden_dim,
            dropout,
        )
        self.predictor = BCBayesianPredictor(
            encoder_output_dim, predictor_hidden_dim, dropout
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.predictor(self.encoder(x))

    def set_mc_dropout_mode(self, active: bool):
        """Toggle dropout layers only, independent of the rest of eval().
        Standard MC-Dropout: keep sampling stochastic masks at inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(active)


class BCPhysicsCalibrator(nn.Module):
    """s_psi(t, x): small residual DNN bridging the idealized physics model
    G_theta and real-world behaviour (paper Eq. 12-14). Deliberately shallow
    and narrow -- it only needs to model a residual, not the full dynamics."""

    def __init__(self, input_dim, hidden_dim=32, layers_num=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers += [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
            elif i == layers_num - 1:
                layers += [nn.Linear(hidden_dim, 1)]
            else:
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, xt):
        return self.net(xt)


class BCPINN(nn.Module):
    """BCPINN architecture and model-specific mathematics."""

    def __init__(self, args, seed=None, quiet=False):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.args = args
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.relu = nn.ReLU()
        self.solution_u = BCBayesianSolutionU(
            input_dim=args.input_dim,
            encoder_output_dim=args.encoder_output_dim,
            encoder_layers_num=args.encoder_layers_num,
            encoder_hidden_dim=args.encoder_hidden_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            dropout=args.dropout,
        ).to(self.device)
        self.dynamical_F = BCMLP(
            input_dim=args.F_input_dim,
            output_dim=args.F_output_dim,
            layers_num=args.F_layers_num,
            hidden_dim=args.F_hidden_dim,
            dropout=args.dropout,
        ).to(self.device)
        self.calibrator = BCPhysicsCalibrator(
            input_dim=args.input_dim,
            hidden_dim=args.calibrator_hidden_dim,
            layers_num=args.calibrator_layers_num,
            dropout=args.calibrator_dropout,
        ).to(self.device)

    def compute_calibrated_residual(self, xt):
        """R = du/dt - G_theta(t,x,u,u_x,u_t) - s_psi(t,x)"""
        xt = xt.detach().requires_grad_(True)
        u = self.solution_u(xt)
        grads = grad(
            outputs=u,
            inputs=xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = grads[:, :-1]
        u_t = grads[:, -1:]
        rhs_input = torch.cat([xt, u, u_x, u_t], dim=1)
        f_hat = self.dynamical_F(rhs_input)
        s_psi = self.calibrator(xt)
        R = u_t - f_hat - s_psi
        return (u, R)

    def forward(self, x):
        return self.solution_u(x)

    def compute_monotonicity_loss(self, u1, u2, y1, y2):
        """
        EXACT monotonicity term used by this project's own models:

            IAW-PI-EKAN:  relu(torch.mul(u2 - u1, y1 - y2)).mean()
            PINN.py:      relu(torch.mul(u2 - u1, y1 - y2)).sum()

        Penalises predicted SOH moving in the opposite direction to the
        measured SOH between consecutive cycles. Reduction is selected by
        --mono_reduction ('mean' matches IAW-PI-EKAN, 'sum' matches PINN.py).
        """
        penalty = self.relu(torch.mul(u2 - u1, y1 - y2))
        return penalty.sum() if self.args.mono_reduction == "sum" else penalty.mean()

    @torch.no_grad()
    def predict_with_uncertainty(self, x, T=None):
        T = T or self.args.mc_samples
        self.solution_u.eval()
        self.solution_u.set_mc_dropout_mode(True)
        x = x.to(self.device)
        samples = torch.stack([self.solution_u(x) for _ in range(T)], dim=0)
        mu = samples.mean(dim=0)
        var = samples.var(dim=0, unbiased=False)
        self.solution_u.set_mc_dropout_mode(False)
        return (mu, var)


class SEQKANLinear(nn.Module):

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
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(
                    -spline_order, grid_size + spline_order + 1, dtype=torch.float32
                )
                * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("spline_scaler", None)
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.grid_size + 1,
                        self.in_features,
                        self.out_features,
                        device=self.grid.device,
                    )
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order], noise
            )
            self.spline_weight.data.copy_(
                (1.0 if self.enable_standalone_scale_spline else self.scale_spline)
                * coeff
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x, y):
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class LSTMModel(nn.Module):
    """Baseline 1: plain LSTM."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, predictor_hidden):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, predictor_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        last = out[:, -1, :]
        return self.predictor(last)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerModel(nn.Module):
    """
    Baseline 2: Transformer encoder over the real W-length window, pooled at
    the last time step. (Simplified from the original encoder-decoder design,
    which used a single learned query token to compensate for a 1-token
    encoder; with genuine multi-token sequences an encoder-only design with
    last-token pooling is the standard, simpler choice and keeps parameter
    count comparable to LSTM/LSTM-KAN for a fair comparison.)
    """

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        predictor_hidden,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, predictor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

    def forward(self, x):
        src = self.input_embedding(x) * self.d_model**0.5 * 0.1
        src = self.pos_encoding(src)
        memory = self.encoder(src)
        pooled = self.layer_norm(memory[:, -1, :])
        return self.predictor(pooled)


class LSTMKANModel(nn.Module):
    """Baseline 3: LSTM backbone + KANLinear head (plain KAN, not EKAN)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        kan_grid_size=5,
        kan_spline_order=3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.kan_head = SEQKANLinear(
            hidden_dim, 1, grid_size=kan_grid_size, spline_order=kan_spline_order
        )

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.kan_head(last)


class RNNKANModel(nn.Module):
    """Vanilla tanh RNN backbone with a SiLU-based spline KAN head."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        kan_grid_size=5,
        kan_spline_order=3,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.kan_head = SEQKANLinear(
            hidden_dim, 1, grid_size=kan_grid_size, spline_order=kan_spline_order
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = self.dropout(out[:, -1, :])
        return self.kan_head(last)


def build_baseline(model_name, input_dim, hp):
    if model_name == "LSTM":
        return LSTMModel(
            input_dim,
            hp["hidden_dim"],
            hp["num_layers"],
            hp["dropout"],
            hp["predictor_hidden"],
        )
    if model_name == "Transformer":
        return TransformerModel(
            input_dim,
            hp["hidden_dim"],
            hp["nhead"],
            hp["num_layers"],
            hp["hidden_dim"] * 4,
            hp["dropout"],
            hp["predictor_hidden"],
        )
    if model_name == "LSTM-KAN":
        return LSTMKANModel(
            input_dim,
            hp["hidden_dim"],
            hp["num_layers"],
            hp["dropout"],
            hp["kan_grid_size"],
            hp["kan_spline_order"],
        )
    if model_name == "RNN-KAN":
        return RNNKANModel(
            input_dim,
            hp["hidden_dim"],
            hp["num_layers"],
            hp["dropout"],
            hp["kan_grid_size"],
            hp["kan_spline_order"],
        )
    raise ValueError(f"Unknown model_name: {model_name}")
