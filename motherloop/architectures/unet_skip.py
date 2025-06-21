# architectures/unet_skip.py
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

class UNetBlock(nn.Module):
    """U-Net style block with skip connections"""
    def __init__(self, in_features, out_features, activation):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(out_features, out_features)
        self.activation = activation
        
        # Skip connection (if dimensions match)
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        return self.activation(out + identity)

class UNetSkipPINN(nn.Module):
    """U-Net inspired PINN with skip connections"""
    
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
        
        # Encoder (downsampling)
        self.encoder1 = UNetBlock(fourier_size, 128, self.activation)
        self.encoder2 = UNetBlock(128, 256, self.activation)
        self.encoder3 = UNetBlock(256, 512, self.activation)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 512, self.activation)
        
        # Decoder (upsampling) with skip connections
        self.decoder3 = UNetBlock(512 + 512, 256, self.activation)  # +skip from encoder3
        self.decoder2 = UNetBlock(256 + 256, 128, self.activation)  # +skip from encoder2
        self.decoder1 = UNetBlock(128 + 128, 64, self.activation)   # +skip from encoder1
        
        # Output layer
        self.output_layer = nn.Linear(64, output_dim)
        
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
        
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder path with skip connections
        d3 = self.decoder3(torch.cat([b, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Output
        return self.output_layer(d1)