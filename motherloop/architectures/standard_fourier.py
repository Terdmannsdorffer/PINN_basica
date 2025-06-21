# architectures/standard_fourier.py
import torch
import torch.nn as nn
import numpy as np

class Mish(nn.Module):
    """Mish activation function"""
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping for better coordinate encoding"""
    def __init__(self, input_dim=2, mapping_size=256, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size // 2) * scale)
    
    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class StandardFourierPINN(nn.Module):
    """Standard PINN with Fourier features and configurable activation"""
    
    def __init__(self, input_dim=2, output_dim=3, hidden_layers=[128, 128, 128, 128], 
                 fourier_size=256, fourier_scale=10.0, activation='tanh'):
        super().__init__()
        
        # Fourier feature mapping
        self.fourier_layer = FourierFeatureMapping(input_dim, fourier_size, fourier_scale)
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = Swish()
        elif activation == 'mish':
            self.activation = Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Network layers
        layers = []
        in_features = fourier_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                self.activation
            ])
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fourier_layer(x)
        return self.network(x)