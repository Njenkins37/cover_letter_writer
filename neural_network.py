"""
Deep Linear Neural Network
==========================
A fully configurable feed-forward network built entirely from linear (fully
connected) layers.  All hyperparameters are constructor arguments so the same
class works for tabular classification, regression, and any other task where
the dataset can be represented as a flat feature vector.

Default hidden architecture (8 layers):
    input → 2048 → 1024 → 512 → 256 → 128 → 64 → 32 → output

Each hidden layer is wrapped in a LinearBlock:
    Linear → BatchNorm1d → ReLU → Dropout

Usage
-----
    from neural_network import DeepLinearNetwork

    model = DeepLinearNetwork(input_size=784, output_size=10)          # MNIST
    model = DeepLinearNetwork(input_size=30, output_size=1,            # binary
                              task="regression")
    model = DeepLinearNetwork(input_size=128, output_size=5,           # custom
                              hidden_sizes=[512, 512, 256, 128, 64])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class LinearBlock(nn.Module):
    """Linear layer followed by BatchNorm, ReLU, and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DeepLinearNetwork(nn.Module):
    """Deep feed-forward network composed entirely of linear layers.

    Parameters
    ----------
    input_size:
        Number of input features (must match your dataset's feature dimension).
    output_size:
        Number of output units.  For binary classification use 1; for
        multi-class use the number of classes; for regression use 1.
    hidden_sizes:
        List of hidden-layer widths.  If *None* the default 8-layer
        tapering architecture is used:
        [2048, 1024, 512, 256, 128, 64, 32].
    dropout:
        Dropout probability applied after every hidden layer.
    task:
        ``"classification"`` – no activation on the output layer (use
        CrossEntropyLoss or BCEWithLogitsLoss in your training loop).
        ``"regression"``     – no activation on the output layer either, but
        the convention signals to callers which loss to use (MSELoss, etc.).
    """

    # Default deep tapering architecture (8 linear layers total)
    _DEFAULT_HIDDEN: List[int] = [2048, 1024, 512, 256, 128, 64, 32]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
        task: str = "classification",
    ):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError('task must be "classification" or "regression"')

        self.input_size = input_size
        self.output_size = output_size
        self.task = task

        sizes = hidden_sizes if hidden_sizes is not None else self._DEFAULT_HIDDEN

        # Build hidden layers
        layers: List[nn.Module] = []
        prev = input_size
        for width in sizes:
            layers.append(LinearBlock(prev, width, dropout=dropout))
            prev = width

        self.hidden = nn.Sequential(*layers)

        # Output head — no activation; caller's loss function handles that
        self.output_head = nn.Linear(prev, output_size)

        # Weight initialisation (He uniform for ReLU networks)
        self._init_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(batch, input_size)``.

        Returns:
            Raw logits / predictions of shape ``(batch, output_size)``.
        """
        x = self.hidden(x)
        return self.output_head(x)

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        """Print a human-readable architecture summary."""
        print(f"\nDeepLinearNetwork  ({self.task})")
        print(f"  Input size  : {self.input_size}")
        for i, block in enumerate(self.hidden):
            lin = block.block[0]          # first module inside LinearBlock
            print(f"  Hidden {i + 1:>2}   : {lin.in_features} → {lin.out_features}")
        print(f"  Output      : {self.output_head.in_features} → {self.output_size}")
        print(f"  Parameters  : {self.count_parameters():,}")
        print()
