import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from util.util import set_seed


class PISin(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class PIKANLinear(nn.Module):

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
        base_activation=PISin,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
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
                    - 0.5
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
        grid = self.grid
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
        act_loss = l1_fake.sum()
        p = l1_fake / act_loss
        ent_loss = -torch.sum(p * p.log())
        return regularize_activation * act_loss + regularize_entropy * ent_loss


class PIEfficientKAN(nn.Module):

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=PISin,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList(
            [
                PIKANLinear(
                    in_f,
                    out_f,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
                for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
            ]
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


class PIEKAN(nn.Module):
    """PIEKAN architecture and model-specific mathematics."""

    def __init__(self, args, seed=None):
        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.args = args
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.solution_u = PIEfficientKAN(
            layers_hidden=args.kan_hidden_layers,
            grid_size=args.kan_grid_size,
            spline_order=args.kan_spline_order,
            scale_noise=args.kan_scale_noise,
            scale_base=args.kan_scale_base,
            scale_spline=args.kan_scale_spline,
            base_activation=PISin,
            grid_eps=args.kan_grid_eps,
            grid_range=args.kan_grid_range,
        ).to(self.device)
        f_layers = [args.F_input_dim]
        for _ in range(args.F_layers_num - 1):
            f_layers.append(args.F_hidden_dim)
        f_layers.append(args.F_output_dim)
        self.dynamical_F = PIEfficientKAN(
            layers_hidden=f_layers,
            grid_size=args.F_kan_grid_size,
            spline_order=args.F_kan_spline_order,
            scale_noise=args.F_kan_scale_noise,
            scale_base=args.F_kan_scale_base,
            scale_spline=args.F_kan_scale_spline,
            base_activation=PISin,
            grid_eps=args.F_kan_grid_eps,
            grid_range=args.F_kan_grid_range,
        ).to(self.device)
        _init = torch.log(torch.tensor(args.sigma_init**2, dtype=torch.float32)).item()
        self.log_sigma_squared_data = nn.Parameter(
            torch.full((), _init, dtype=torch.float32).to(self.device)
        )
        self.log_sigma_squared_pde = nn.Parameter(
            torch.full((), _init, dtype=torch.float32).to(self.device)
        )
        self.log_sigma_squared_mono = nn.Parameter(
            torch.full((), _init, dtype=torch.float32).to(self.device)
        )
        self.gamma_inv = 1.0 / args.gamma
        self.relu = nn.ReLU()

    def compute_adaptive_weights(self):
        sigma2_data = torch.exp(self.log_sigma_squared_data)
        sigma2_pde = torch.exp(self.log_sigma_squared_pde)
        sigma2_mono = torch.exp(self.log_sigma_squared_mono)
        lambda_data = 1.0 / (sigma2_data + self.gamma_inv)
        lambda_pde = 1.0 / (sigma2_pde + self.gamma_inv)
        lambda_mono = 1.0 / (sigma2_mono + self.gamma_inv)
        return (
            lambda_data,
            lambda_pde,
            lambda_mono,
            sigma2_data,
            sigma2_pde,
            sigma2_mono,
        )

    def compute_adaptive_loss(self, data_loss, pde_loss, mono_loss):
        lambda_data, lambda_pde, lambda_mono, sigma2_data, sigma2_pde, sigma2_mono = (
            self.compute_adaptive_weights()
        )
        eps = 1e-08
        total_loss = (
            lambda_data * data_loss
            + lambda_pde * pde_loss
            + lambda_mono * mono_loss
            + torch.log(sigma2_data + self.gamma_inv + eps)
            + torch.log(sigma2_pde + self.gamma_inv + eps)
            + torch.log(sigma2_mono + self.gamma_inv + eps)
        )
        return (total_loss, lambda_data, lambda_pde, lambda_mono)

    def compute_pde_residual(self, xt):
        """
        Compute the integer-order PDE residual  f = ∂u/∂t − G(t,x,u,u_t,u_x).

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
