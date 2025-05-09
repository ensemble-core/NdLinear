import torch
import torch.nn as nn
from typing import Literal, Tuple, Optional


class NdLinearGated(nn.Module):
    """
    NdLinearGated: A PyTorch layer for projecting tensors into multi-space representations with gating mechanisms.
    
    Extends the NdLinear concept by incorporating gating mechanisms that control information flow.
    This allows the model to selectively utilize transformations based on input characteristics,
    enabling more adaptive and context-dependent multi-space representations.
    """
    def __init__(self, 
                 input_dims: tuple, 
                 hidden_size: tuple, 
                 transform_outer: bool = True,
                 gating_mode: Literal["soft", "hard"] = "soft",
                 gating_hidden_dim: int = 16,
                 gated_modes: Literal["all", "first", "topk"] = "all") -> None:
        """
        Initialize the NdLinearGated layer.
        
        Args:
            input_dims: Shape of input tensor (excluding batch dimension).
            hidden_size: Target hidden dimensions after transformation.
            transform_outer: If True, transforms from outer to inner dimensions.
            gating_mode: Type of gating mechanism - "soft" uses continuous values, "hard" uses binary.
            gating_hidden_dim: Hidden dimension size for the gating networks.
            gated_modes: Specifies which dimensions to apply gating to.
        """
        super(NdLinearGated, self).__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_layers = len(input_dims)
        self.transform_outer = transform_outer
        self.gating_mode = gating_mode
        self.gating_hidden_dim = gating_hidden_dim
        self.gated_modes = gated_modes
        
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
        
        self.topk_modes = None
        self.first_batch_processed = False
        
    def _compute_topk_modes(self, X: torch.Tensor) -> list:
        mode_stds = []
        for i in range(self.num_layers):
            transpose_dim = i + 1 if self.transform_outer else self.num_layers - i
            X_transposed = torch.transpose(X, transpose_dim, self.num_layers)
            X_mean = X_transposed.mean(dim=tuple(range(len(X_transposed.shape) - 1)))
            mode_stds.append(X_mean.std().item())
        
        sorted_modes = sorted(range(len(mode_stds)), key=lambda i: mode_stds[i], reverse=True)
        return sorted_modes[:2] 
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to project input tensor into a new multi-space representation with gating.
        
        Applies dimensional transformations with selective gating based on the configured mode.
        The gating mechanism allows the network to adaptively choose between transformed
        representations and identity mappings.
        
        Args:
            X (torch.Tensor): Input tensor with shape [batch_size, *input_dims]
            
        Returns:
            torch.Tensor: Output tensor with shape [batch_size, *hidden_size]
        """
        num_transforms = self.num_layers
        
        if self.gated_modes == "topk" and not self.first_batch_processed:
            self.topk_modes = self._compute_topk_modes(X)
            self.first_batch_processed = True
            
        for i in range(num_transforms):
            if self.transform_outer:
                layer_idx = i
                transpose_dim = i + 1
            else:
                layer_idx = num_transforms - (i+1)
                transpose_dim = num_transforms - i
                
            apply_gating = False
            if self.gated_modes == "all":
                apply_gating = True
            elif self.gated_modes == "first" and i == 0:
                apply_gating = True
            elif self.gated_modes == "topk" and self.topk_modes and layer_idx in self.topk_modes:
                apply_gating = True
                
            X_original = X.clone()
            
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()
            
            X_size = X.shape[:-1]
            
            X_flat = X.view(-1, X.shape[-1])
            
            X_transformed = self.align_layers[layer_idx](X_flat)
            
            if apply_gating:
                X_mean = X_flat.mean(dim=0, keepdim=True)
                
                gate = self.gate_networks[layer_idx](X_mean)
                
                X_transformed = X_transformed.view(*X_size, X_transformed.shape[-1])
                
                X_identity = torch.transpose(X_original, transpose_dim, num_transforms).contiguous()
                
                X_identity_flat = X_identity.view(-1, X_identity.shape[-1])
                
                if X_transformed.shape[-1] != X_identity_flat.shape[-1]:
                    identity_flat = self.identity_projections[layer_idx](X_identity_flat)
                else:
                    identity_flat = X_identity_flat
                
                if self.gating_mode == "soft":
                    X_flat = gate * X_transformed.view(-1, X_transformed.shape[-1]) + (1 - gate) * identity_flat
                else: 
                    X_flat = torch.where(gate > 0.5, 
                                         X_transformed.view(-1, X_transformed.shape[-1]),
                                         identity_flat)
                    
                X = X_flat.view(*X_size, X_flat.shape[-1])
            else:
                X = X_transformed.view(*X_size, X_transformed.shape[-1])
            
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()
            
        return X

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(input_dims={self.input_dims}, "
                f"hidden_size={self.hidden_size}, transform_outer={self.transform_outer}, "
                f"gating_mode={self.gating_mode}, gating_hidden_dim={self.gating_hidden_dim}, "
                f"gated_modes={self.gated_modes})") 