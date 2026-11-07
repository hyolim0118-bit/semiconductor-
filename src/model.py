"""PyTorch multi-output thickness regression baseline (no training)."""

import torch
from torch import nn

from .config import HIDDEN_DIMS, INPUT_DIM, OUTPUT_DIM


class MLPRegressor(nn.Module):
    """Map standardized reflectance to layer_1 through layer_4, in order."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS,
                 output_dim=OUTPUT_DIM):
        super().__init__()
        dimensions = (input_dim,) + tuple(hidden_dims) + (output_dim,)
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
               for size in dimensions):
            raise ValueError("All layer dimensions must be positive integers")
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        for index, (in_features, out_features) in enumerate(
                zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Accept a two-dimensional Tensor of shape (batch_size, input_dim)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected input to be a torch.Tensor")
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                "Expected input shape (batch_size, {}), got {}".format(
                    self.input_dim, tuple(x.shape)))
        return self.network(x)


def count_trainable_parameters(model):
    """Count weights and biases that require gradients."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


if __name__ == "__main__":
    model = MLPRegressor()
    print(model)
    print("Trainable parameters: {:,}".format(count_trainable_parameters(model)))
