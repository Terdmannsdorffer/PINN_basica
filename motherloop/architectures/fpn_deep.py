# architectures/fpn_deep.py
import torch
import torch.nn as nn

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=256, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size // 2) * scale)
    
    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    """Residual block with optional bottleneck"""
    def __init__(self, features, activation, use_bottleneck=False):
        super().__init__()
        
        if use_bottleneck:
            # Bottleneck design: reduce -> process -> expand
            mid_features = features // 4
            self.layers = nn.Sequential(
                nn.Linear(features, mid_features),
                activation,
                nn.Linear(mid_features, mid_features),
                activation,
                nn.Linear(mid_features, features)
            )
        else:
            # Standard residual
            self.layers = nn.Sequential(
                nn.Linear(features, features),
                activation,
                nn.Linear(features, features)
            )
        
        self.activation = activation
    
    def forward(self, x):
        return self.activation(x + self.layers(x))

class FPNDeepPINN(nn.Module):
    """Feature Pyramid Network inspired PINN with deep connections"""
    
    def __init__(self, input_dim=2, output_dim=3, fourier_size=256, 
                 fourier_scale=10.0, activation='tanh'):
        super().__init__()
        
        # Fourier features
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
        
        # Multi-scale feature extraction (like FPN)
        self.level1 = nn.Sequential(
            nn.Linear(fourier_size, 128),
            self.activation,
            ResidualBlock(128, self.activation),
            ResidualBlock(128, self.activation)
        )
        
        self.level2 = nn.Sequential(
            nn.Linear(128, 256),
            self.activation,
            ResidualBlock(256, self.activation, use_bottleneck=True),
            ResidualBlock(256, self.activation, use_bottleneck=True)
        )
        
        self.level3 = nn.Sequential(
            nn.Linear(256, 512),
            self.activation,
            ResidualBlock(512, self.activation, use_bottleneck=True),
            ResidualBlock(512, self.activation, use_bottleneck=True)
        )
        
        # Deep processing
        self.deep_layers = nn.Sequential(
            ResidualBlock(512, self.activation, use_bottleneck=True),
            ResidualBlock(512, self.activation, use_bottleneck=True)
        )
        
        # Lateral connections (FPN style) - simplified
        self.lateral3 = nn.Linear(512, 256)  # f3 -> 256
        self.lateral2 = nn.Linear(256, 128)  # f2 -> 128  
        self.lateral1 = nn.Linear(128, 128)  # f1 -> 128 (identity-like)
        
        # Top-down pathway
        self.topdown3 = nn.Linear(512, 256)  # deep_f3 -> 256
        self.topdown2 = nn.Linear(256, 128)  # -> 128
        
        # Feature fusion - simplified to use final_features
        self.fusion = nn.Sequential(
            nn.Linear(128, 256),  # Work with final fused features
            self.activation,
            ResidualBlock(256, self.activation),
            nn.Linear(256, 128),
            self.activation
        )
        
        # Output layer
        self.output_layer = nn.Linear(128, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Fourier encoding
        x = self.fourier_layer(x)
        
        # Bottom-up pathway
        f1 = self.level1(x)      # 128 features
        f2 = self.level2(f1)     # 256 features  
        f3 = self.level3(f2)     # 512 features
        
        # Deep processing at highest level
        deep_f3 = self.deep_layers(f3)
        
        # Top-down pathway with lateral connections (FPN style)
        # Level 3: Combine deep processing with f3
        td3 = self.topdown3(deep_f3)   # 512 -> 256
        lat3 = self.lateral3(f3)       # 512 -> 256
        fused_f2 = td3 + lat3          # 256 + 256 = 256
        
        # Level 2: Combine with f2
        td2 = self.topdown2(fused_f2)  # 256 -> 128  
        lat2 = self.lateral2(f2)       # 256 -> 128
        fused_f1 = td2 + lat2          # 128 + 128 = 128
        
        # Level 1: Add f1 (already 128)
        lat1 = self.lateral1(f1)       # 128 -> 128 
        final_features = fused_f1 + lat1  # 128 + 128 = 128
        
        # Feature fusion and output
        fused = self.fusion(final_features)
        
        # Output
        return self.output_layer(fused)