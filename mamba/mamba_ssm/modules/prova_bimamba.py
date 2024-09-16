### Prova per vedere se Bi-Mamba funziona già dentro videomamba

#import torch
import torchaudio

print('done')

# from mamba_ssm import Mamba

# # Creo un dataset sintetico di audio
# batch, length, dim = 32, 41056, 1 
# x = torch.randn(batch, length, dim).to("cuda")

# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model = dim, # Model dimension d_model
#     d_state = 16,  # SSM state expansion factor
#     d_conv = 4,    # Local convolution width
#     expand = 2,    # Block expansion factor
#     bimamba = False
# ).to("cuda")
# y = model(x)

# print(y.shape)
