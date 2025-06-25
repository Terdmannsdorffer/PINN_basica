# model.py - ENHANCED VERSION with Velocity Scaling
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


class EnhancedDeepPINN(nn.Module):
    def __init__(
            self, input_dim=2, 
            output_dim=3, 
            hidden_layers=[128, 128, 128, 128, 128], 
            fourier_mapping_size=256, 
            fourier_scale=10.0, 
            activation='swish', 
            beta=1.0,
            piv_velocity_stats=None  # {'u_mean': float, 'v_mean': float, 'mag_mean': float, 'mag_std': float}
        ):
        """Enhanced Physics-Informed Neural Network with velocity scaling and magnitude matching."""
        super(EnhancedDeepPINN, self).__init__()
        
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
        
        # Output layer (produces normalized velocities and pressure)
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
        # ENHANCEMENT 1: Learnable velocity scaling parameters
        self.u_scale = nn.Parameter(torch.tensor(1.0))  # Horizontal velocity scale
        self.v_scale = nn.Parameter(torch.tensor(1.0))  # Vertical velocity scale
        self.global_scale = nn.Parameter(torch.tensor(1.0))  # Global velocity scale
        
        # ENHANCEMENT 2: Pressure scaling
        self.pressure_scale = nn.Parameter(torch.tensor(1.0))
        
        # Store PIV statistics for initialization and calibration
        self.piv_stats = piv_velocity_stats
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        # ENHANCEMENT 3: Initialize with PIV statistics if available
        if piv_velocity_stats is not None:
            self._initialize_with_piv_statistics()
    
    def _initialize_weights(self):
        """Xavier initialization for better training convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _initialize_with_piv_statistics(self):
        """ENHANCEMENT 4: Initialize network to produce PIV-like velocities"""
        if self.piv_stats is None:
            return
            
        with torch.no_grad():
            # Initialize velocity scaling parameters
            self.u_scale.data = torch.tensor(abs(self.piv_stats.get('u_mean', 0.005)))
            self.v_scale.data = torch.tensor(abs(self.piv_stats.get('v_mean', 0.005)))
            self.global_scale.data = torch.tensor(self.piv_stats.get('mag_mean', 0.005))
            
            # Initialize output layer bias to produce realistic velocities
            self.output_layer.bias[0] = 0.0  # u-component (will be scaled)
            self.output_layer.bias[1] = -1.0  # v-component (negative for downward flow)
            self.output_layer.bias[2] = 0.0  # pressure
            
            # Scale output weights to reasonable range
            self.output_layer.weight[:2, :] *= 0.1  # Smaller initial velocity weights
            
        print(f"Initialized with PIV statistics:")
        print(f"  u_scale: {self.u_scale.item():.6f}")
        print(f"  v_scale: {self.v_scale.item():.6f}")
        print(f"  global_scale: {self.global_scale.item():.6f}")
    
    def forward(self, x):
        # Apply Fourier feature mapping
        x = self.fourier_layer(x)
        
        # Apply input layer
        x = self.input_layer(x)
        
        # Apply hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Apply output layer (produces normalized outputs)
        raw_output = self.output_layer(x)
        
        # ENHANCEMENT 5: Apply velocity scaling
        u_raw, v_raw, p_raw = raw_output[:, 0:1], raw_output[:, 1:2], raw_output[:, 2:3]
        
        # Scale velocities with learnable parameters
        u_scaled = u_raw * self.u_scale * self.global_scale
        v_scaled = v_raw * self.v_scale * self.global_scale
        p_scaled = p_raw * self.pressure_scale
        # Clamp escalas para evitar inversión de signo
        self.u_scale.data.clamp_(min=1e-5)
        self.v_scale.data.clamp_(min=1e-5)
        self.global_scale.data.clamp_(min=1e-5)

        return torch.cat([u_scaled, v_scaled, p_scaled], dim=1)
    
    def get_velocity_scales(self):
        """Get current velocity scaling parameters"""
        return {
            'u_scale': self.u_scale.item(),
            'v_scale': self.v_scale.item(), 
            'global_scale': self.global_scale.item(),
            'effective_u_scale': (self.u_scale * self.global_scale).item(),
            'effective_v_scale': (self.v_scale * self.global_scale).item()
        }
    
    def set_velocity_scales(self, u_scale=None, v_scale=None, global_scale=None):
        """Manually set velocity scaling parameters"""
        with torch.no_grad():
            if u_scale is not None:
                self.u_scale.data = torch.tensor(float(u_scale))
            if v_scale is not None:
                self.v_scale.data = torch.tensor(float(v_scale))
            if global_scale is not None:
                self.global_scale.data = torch.tensor(float(global_scale))


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        return x  # Identity residual connection


# Compatibility alias for backward compatibility
DeepPINN = EnhancedDeepPINN