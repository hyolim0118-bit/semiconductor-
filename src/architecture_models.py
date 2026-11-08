"""Project-specific MLP and pre-normalized residual candidates; baseline unchanged."""

import torch
from torch import nn

from .model import MLPRegressor


def check_input(x, input_dim):
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if x.ndim != 2 or x.shape[1] != input_dim:
        raise ValueError("Expected shape (batch_size, {})".format(input_dim))


class ComparisonMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        check_input(x, self.input_dim)
        return self.network(x)


class ResidualBlock(nn.Module):
    """BN -> Linear -> ReLU -> Dropout -> Linear -> Dropout, then add input."""

    def __init__(self, width, hidden_factor=1.0, hidden_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        hidden = int(width * hidden_factor)
        if width <= 0 or hidden <= 0:
            raise ValueError("Residual widths must be positive")
        self.branch = nn.Sequential(
            nn.BatchNorm1d(width), nn.Linear(width, hidden), nn.ReLU(),
            nn.Dropout(hidden_dropout), nn.Linear(hidden, width), nn.Dropout(residual_dropout))

    def forward(self, x):
        return x + self.branch(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, width, blocks, hidden_factor=1.0,
                 hidden_dropout=0.1, residual_dropout=0.1, head_batchnorm=True):
        super().__init__()
        self.input_dim = input_dim
        if blocks < 1:
            raise ValueError("At least one residual block is required")
        self.input_layer = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, hidden_factor, hidden_dropout, residual_dropout)
            for _ in range(blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(width) if head_batchnorm else nn.Identity(),
                                  nn.ReLU(), nn.Linear(width, output_dim))

    def forward(self, x):
        check_input(x, self.input_dim)
        return self.head(self.blocks(self.input_layer(x)))


def build_model(spec, input_dim=226, output_dim=4):
    if spec["kind"] == "mlp":
        # A uses the original class and initialization order, without any migration.
        if spec["id"] == "A":
            if spec["dropout"] != 0:
                raise ValueError("Baseline A preserves the original zero-dropout MLP")
            return MLPRegressor(input_dim, tuple(spec["hidden_dims"]), output_dim)
        return ComparisonMLP(input_dim, spec["hidden_dims"], output_dim, spec["dropout"])
    if spec["kind"] == "residual":
        return ResidualMLP(input_dim, output_dim, **{
            key: spec[key] for key in ("width", "blocks", "hidden_factor", "hidden_dropout",
                                      "residual_dropout", "head_batchnorm")})
    raise ValueError("Unknown model kind: " + spec["kind"])
