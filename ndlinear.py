
import torch
import torch.nn as nn
import torch.optim as optim

class NdLinear(nn.Module):
    def __init__(self, input_dims: tuple, hidden_size: tuple, transform_outer=True, bias=True):
        """
        NdLinear: Multidimensional Linear Projection Layer.
        Projects tensor inputs across multiple dimensions independently.

        Args:
            input_dims (tuple): Shape of the input excluding batch dimension.
            hidden_size (tuple): Shape after transformation (must be same length as input_dims).
            transform_outer (bool): Whether to transform outer dims first (True) or inner (False).
            bias (bool): Whether to use bias in each Linear layer.
        """
        super().__init__()
        if len(input_dims) != len(hidden_size):
            raise ValueError("Input and hidden shape must have the same rank.")
        
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_layers = len(input_dims)
        self.transform_outer = transform_outer

        self.align_layers = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i], bias=bias)
            for i in range(self.num_layers)
        ])

    def forward(self, X):
        transform_indices = range(self.num_layers) if self.transform_outer else reversed(range(self.num_layers))

        for i in transform_indices:
            layer = self.align_layers[i]
            transpose_dim = i + 1  # +1 to skip batch

            X = torch.transpose(X, transpose_dim, self.num_layers).contiguous()
            orig_shape = X.shape[:-1]
            X = X.view(-1, X.shape[-1])
            X = layer(X)
            X = X.view(*orig_shape, X.shape[-1])
            X = torch.transpose(X, transpose_dim, self.num_layers).contiguous()

        assert X.shape[1:] == self.hidden_size, f"Expected shape {self.hidden_size}, got {X.shape[1:]}"
        return X
