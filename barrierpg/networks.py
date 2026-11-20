"""Optional PyTorch backbones for vector observations."""
from __future__ import annotations

from torch import Tensor, nn


class _FeedForwardBlock(nn.Module):
    def __init__(self, width: int, hidden_width: int, *, residual: bool, layer_norm: bool,
                 activation_fn: type[nn.Module]):
        super().__init__()
        self.residual = residual
        self.branch = nn.Sequential(
            nn.LayerNorm(width) if layer_norm else nn.Identity(),
            nn.Linear(width, hidden_width),
            activation_fn(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, features: Tensor) -> Tensor:
        update = self.branch(features)
        return features + update if self.residual else update


class ResidualMLP(nn.Module):
    """Map vectors of size `input_dim` to vectors of size `width`.

    Each block applies a two-layer feed-forward branch to the optionally normalized
    input. The skip connection adds the unnormalized input back to the branch output.
    """

    def __init__(self, input_dim: int, *, width: int = 64, depth: int = 2,
                 hidden_width: int | None = None, residual: bool = True, layer_norm: bool = True,
                 activation_fn: type[nn.Module] = nn.SiLU):
        super().__init__()
        hidden_width = width if hidden_width is None else hidden_width
        if min(input_dim, width, depth, hidden_width) < 1:
            raise ValueError("input_dim, width, depth, and hidden_width must be positive")
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(
            _FeedForwardBlock(width, hidden_width, residual=residual, layer_norm=layer_norm,
                              activation_fn=activation_fn)
            for _ in range(depth)
        ))

    def forward(self, observations: Tensor) -> Tensor:
        return self.blocks(self.input_projection(observations))
