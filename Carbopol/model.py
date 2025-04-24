import torch
import torch.nn as nn

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define a very simple PINN model
class SimplePINN(nn.Module):
    def __init__(self):
        super(SimplePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 3)  # u, v, p outputs
        )
    
    def forward(self, x):
        return self.net(x)