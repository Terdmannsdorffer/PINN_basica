import torch

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constants for L-shaped domain
W = 0.5      # Width of the L shape
L_V = 2.0    # Vertical length of the L
L_H = 3.0    # Horizontal length of the L

# Constants for training
LEARNING_RATE = 0.001
EPOCHS = 2000