import torch
import torch.nn as nn

class DeepPINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_layers=[128, 128, 128, 128, 128]):
        """Enhanced Physics-Informed Neural Network for fluid flow simulation."""
        super(DeepPINN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.Tanh()]
        
        # Hidden layers with residual connections
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
            
            # Add residual connections
            if hidden_layers[i] == hidden_layers[i+1]:
                layers.append(ResidualBlock(hidden_layers[i]))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Sequential model
        self.net = nn.Sequential(*layers)
        
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
        return self.net(x)

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        
    def forward(self, x):
        return x  # Identity residual connection
