import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple


class NdLinearGated(nn.Module):
    def __init__(self, 
                 input_dims: Tuple[int, ...], 
                 hidden_size: Tuple[int, ...], 
                 transform_outer: bool = True,
                 gating_mode: Literal["soft", "hard"] = "soft",
                 gating_hidden_dim: int = 16,
                 gated_modes: Literal["all", "first", "topk", "none"] = "all",
                 topk: int = 2,
                 dropout_p: float = 0.1) -> None:
        super(NdLinearGated, self).__init__()

        if len(input_dims) != len(hidden_size):
            raise ValueError("Input shape and hidden shape must have the same number of dimensions.")

        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_layers = len(input_dims)
        self.transform_outer = transform_outer
        self.gating_mode = gating_mode
        self.gating_hidden_dim = gating_hidden_dim
        self.gated_modes = gated_modes
        self.topk = topk
        self.topk_modes = None
        self.first_batch_processed = False

        self.align_layers = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i]) for i in range(self.num_layers)
        ])

        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], gating_hidden_dim),
                nn.ReLU(),
                nn.Linear(gating_hidden_dim, 1),
                nn.Sigmoid()
            ) for i in range(self.num_layers)
        ])

        self.identity_projections = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i]) if input_dims[i] != hidden_size[i] else nn.Identity()
            for i in range(self.num_layers)
        ])

        self.dropout = nn.Dropout(dropout_p)

    def reset_topk(self, X: torch.Tensor):
        """Recomputes top-k dimension indices based on std of mean values."""
        mode_stds = []
        for i in range(self.num_layers):
            dim = i + 1 if self.transform_outer else self.num_layers - i
            Xt = torch.transpose(X, dim, self.num_layers)
            X_mean = Xt.mean(dim=tuple(range(len(Xt.shape) - 1)))
            mode_stds.append(X_mean.std().item())
        self.topk_modes = sorted(range(self.num_layers), key=lambda i: mode_stds[i], reverse=True)[:self.topk]
        self.first_batch_processed = True

    def _transform_step(self, X: torch.Tensor, layer_idx: int, transpose_dim: int, apply_gating: bool) -> torch.Tensor:
        X_identity = X  # no clone
        X = torch.transpose(X, transpose_dim, self.num_layers).contiguous()
        X_size = X.shape[:-1]
        X_flat = X.view(-1, X.shape[-1])
        X_transformed = self.align_layers[layer_idx](X_flat).view(*X_size, -1)

        if not apply_gating:
            return torch.transpose(X_transformed, transpose_dim, self.num_layers).contiguous()

        # Gating value
        gate = self.gate_networks[layer_idx](X_flat.mean(dim=0, keepdim=True))  # shape [1, 1]
        gate = self.dropout(gate)

        X_identity_transposed = torch.transpose(X_identity, transpose_dim, self.num_layers).contiguous()
        X_identity_flat = X_identity_transposed.view(-1, X_identity_transposed.shape[-1])

        identity_flat = (self.identity_projections[layer_idx](X_identity_flat)
                         if self.input_dims[layer_idx] != self.hidden_size[layer_idx]
                         else X_identity_flat)

        if self.gating_mode == "soft":
            out_flat = gate * X_transformed.view(-1, X_transformed.shape[-1]) + \
                       (1 - gate) * identity_flat
        else:
            out_flat = torch.where(gate > 0.5,
                                   X_transformed.view(-1, X_transformed.shape[-1]),
                                   identity_flat)

        X_out = out_flat.view(*X_size, -1)
        return torch.transpose(X_out, transpose_dim, self.num_layers).contiguous()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.gated_modes == "topk" and not self.first_batch_processed:
            self.reset_topk(X)

        for i in range(self.num_layers):
            if self.transform_outer:
                layer_idx = i
                transpose_dim = i + 1
            else:
                layer_idx = self.num_layers - i - 1
                transpose_dim = self.num_layers - i

            # Determine if gating is applied
            apply_gating = False
            if self.gated_modes == "all":
                apply_gating = True
            elif self.gated_modes == "first" and i == 0:
                apply_gating = True
            elif self.gated_modes == "topk" and self.topk_modes and layer_idx in self.topk_modes:
                apply_gating = True

            X = self._transform_step(X, layer_idx, transpose_dim, apply_gating)

        return X

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(input_dims={self.input_dims}, "
                f"hidden_size={self.hidden_size}, transform_outer={self.transform_outer}, "
                f"gating_mode={self.gating_mode}, gating_hidden_dim={self.gating_hidden_dim}, "
                f"gated_modes={self.gated_modes}, topk={self.topk})")
