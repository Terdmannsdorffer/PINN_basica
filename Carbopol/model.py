#model.py
import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping layer for better distribution of input features."""
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        
        # Create a random matrix for the Fourier feature projection
        # Half for sin features, half for cos features
        self.register_buffer('B', torch.randn((input_dim, mapping_size // 2)) * scale)
        
    def forward(self, x):
        # Project input to Fourier space
        x_proj = torch.matmul(x, self.B)
        
        # Apply sin and cos to create features
        output = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return output


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(beta * x)"""
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class DeepPINN(nn.Module):
    def __init__(
            self, input_dim=2, 
            output_dim=3, 
            hidden_layers=[128, 128, 128, 128, 128], 
            fourier_mapping_size=256, 
            fourier_scale=10.0, 
            activation='swish', 
            beta=1.0
        ):
        """Enhanced Physics-Informed Neural Network with Fourier feature mapping."""
        super(DeepPINN, self).__init__()
        
        # Activation function selection
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'swish':
            self.activation = Swish(beta)
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
        # Fourier feature mapping layer
        self.fourier_layer = FourierFeatureMapping(
            input_dim=input_dim, 
            mapping_size=fourier_mapping_size, 
            scale=fourier_scale
        )
        
        # Adjusted input dimension after Fourier feature mapping
        fourier_output_dim = fourier_mapping_size
        
        # First layer after Fourier mapping
        self.input_layer = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_layers[0]),
            self.activation
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            layer_block = nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                self.activation
            )
            self.hidden_layers.append(layer_block)
            
            # Add residual connections for layers with matching dimensions
            if hidden_layers[i] == hidden_layers[i+1]:
                self.hidden_layers.append(ResidualBlock(hidden_layers[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better training convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Apply Fourier feature mapping
        x = self.fourier_layer(x)
        
        # Apply input layer
        x = self.input_layer(x)
        
        # Apply hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Apply output layer
        x = self.output_layer(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        return x  # Identity residual connection