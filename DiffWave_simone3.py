# This code takes inspiration from DiffWave:
# https://github.com/lmnt-com/diffwave/tree/master

import sys
import numpy as np
from scipy.interpolate import interp1d # for interpolation
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from mamba_ssm import Mamba

# Originally for the spectrogram
import librosa
import librosa.display


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #@simone messo per il train
Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)

def count_parameters(model):
    """
    Count number of parameters, mainly for debugging purposes
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)

    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table

def normalize_audio(audio):
        """
        Normalize an audio signal into [-1, +1].
        It didn't really work. I leave it here for future reference
        """
        return 2 * (audio - audio.min()) / (audio.max() - audio.min()) - 1

# Originally from the authors. I decided to focus only on the raw waveforms

# class SpectrogramUpsampler(nn.Module):
#   def __init__(self, n_mels):
#     super().__init__()
#     self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
#     self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

#   def forward(self, x):
#     x = torch.unsqueeze(x, 1)
#     x = self.conv1(x)
#     x = F.leaky_relu(x, 0.4)
#     x = self.conv2(x)
#     x = F.leaky_relu(x, 0.4)
#     x = torch.squeeze(x, 1)
#     return x
  

def interpolate_audio_signal(audio_signal, scale_factor = 2):
    """
    Linear interpolation on a PyTorch audio tensor. It doubles the length.
    
    :param audio_signal: input audio signal. Expected shape: [batch_size, channels, length].
    :param scale_factor: Scale factor for the (new) length of the signal. Eg: 2 to double the length.
    :return: PyTorch Tensor being the interpolated signal.
    """

    interpolated_signal = F.interpolate(audio_signal, scale_factor=scale_factor, mode='linear', align_corners=False)
    return interpolated_signal



class ResidualBlock(nn.Module):
  def __init__(self, residual_channels, dilation, uncond=False, n_mels = 80):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional (defined by the authors)
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation).to('cuda')
    self.norm_layer = nn.BatchNorm1d(2 * residual_channels)
    
    self.diffusion_projection = Linear(512, residual_channels)
    
    if not uncond: # conditional model
      self.conditioner_projection = Conv1d(residual_channels, 2 * residual_channels, 1).to('cuda') # @simone inventato io
    else: # unconditional model
      self.conditioner_projection = None

    self.residual_channels = residual_channels
    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1).to('cuda')

    self.mamba = Mamba(d_model=self.residual_channels * 2, d_state = 14, d_conv = 4, expand = 2, bimamba = True).to("cuda")



  def forward(self, x, audio_conditioning, diffusion_step, conditioner=None):
    device = 'cuda'
    x = x.to(device)

    # Here the diffusion step is already a vector of shape [1, 512]
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).to(device)
    # Now diffusion_step has shape torch.Size([1, 64, 1]), while x has shape torch.Size([1, 64, 24000])
    
    y = x + diffusion_step # torch.Size([1, 64, 24000])
    y = self.dilated_conv(y) # doubles the channels: torch.Size([1, 128, 24000])
    y = self.norm_layer(y)
    y = y.permute(0, 2, 1).to(device) # [1, 24000, 128]
    batch, length, dim = y.shape # [1, 24000, 128]
    y = self.mamba(y)
    y = y.permute(0, 2, 1) # torch.Size([1, 128, 24000])

    if self.conditioner_projection is None: # using a unconditional model
      
      batch, length, dim = y.shape # @simone [1, 24000, 128]
      
      # I did not focus on unconditional generation
    
    else: # conditioning on the low-resolution audio
      
      # The conditioning audio here has already been interpolated, and therefore it hase shape torch.Size([1, 64, 48000])
      conditioner = self.conditioner_projection(audio_conditioning.to(device)) # torch.Size([1, 128, 24000])
      conditioner = torch.relu(conditioner)
      
      if y.size(-1) != conditioner.size(-1):
        # If y has an odd length less than conditioner, we “cut off” the last value of conditioner.
        # This happens because doubling the size of the conditioner will always make the length even, but perhaps
        # the original 48KhZ audio was of odd length.
        conditioner = conditioner[..., :y.size(-1)]
      y = y + conditioner # torch.Size([1, 128, 24000])
      
    gate, filter = torch.chunk(y, 2, dim=1) # both torch.Size([1, 64, 24000])
    y = torch.sigmoid(gate) * torch.tanh(filter) # torch.Size([1, 64, 16000])

    del gate, filter, diffusion_step, conditioner # to free more GPU memory

    y = self.output_projection(y) # torch.Size([1, 128, 24000])
    residual, skip = torch.chunk(y, 2, dim=1) # torch.Size([1, 64, 24000])

    return (x + residual) / sqrt(2.0), skip # e qui le dimensioni rimangono uguali, sempre entrambi torch.Size([1, 64, 24000]) 


########### Diffusion ###########
def create_noise_schedule(steps=1000, beta_start=1e-6, beta_end=0.006):
    '''From the paper paper Denoising Diffusion Probabilistic Models, Ho et al.'''
    betas = torch.linspace(beta_start, beta_end, steps)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars

def forward_diffusion_sample(x_0, t, noise_schedule):
    '''From the paper Denoising Diffusion Probabilistic Models, Ho et al.,
    page 2, immediately before formula (4).
    It simply add gaussian noise, without learnable parameters'''
    noise = torch.randn_like(x_0) # Creating Gaussian noise
    alpha_bar_t = noise_schedule[t].unsqueeze(-1) # Selecting the value of alpha_bar_t, as in the paper
    
    # Formula (4) of the paper: q(x_t | x_0)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise


def calculate_loss(model, x_0, diffusion_step, noise_schedule, conditioning=None):
    '''
    It takes the original clean audio as input. Adds noise to it. Passes the noisy audio to the model.
    Gets a prediction of the noise. Makes the difference (MSE) and returns the loss between true_noise and predicted_noise
    '''

    # Start from x_0 and add noise with the function we previsouly defined
    x_t, true_noise = forward_diffusion_sample(x_0, diffusion_step, noise_schedule)

    # Then we pass x_t, the noisy version, inside DiffWave to predict the noise
    predicted_noise = model(x_t, diffusion_step, conditioning)

    # Finally we compute the difference, aka loss, between the noise predicted
    #  by the model and the original one, deinfed in formula (4)
    loss = F.mse_loss(predicted_noise, true_noise)
    return loss


#################################


class DiffWave(nn.Module):
    """
    The final, complete model
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.input_projection = Conv1d(1, params.residual_channels, 1) # prende l'audio [1, 1, 48000] e lo rende [1, 64, 48000]

        self.conditioning_projection = Conv1d(1, params.residual_channels, 1) # @simone inventato io
        self.conditioning_norm = nn.BatchNorm1d(params.residual_channels) # così in teoria evitiamo che si propaghi il rumore iniziale

        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        

        self.noise_schedule = create_noise_schedule(1000)

    def forward(self, x, diffusion_step, audio_conditioning):
        '''
        Here:
        - x is the 48KhZ input audio, to which noise will be added
        - diffusion_step is the timestep for the diffusion
        - audio conditioinng is the downsampled audio, i.e. of half length, which will be interpolated to match the length of x
        '''

        device = x.device

        '''Processing of the input'''
        if x.dim() == 2:
          x = x.unsqueeze(1)
        x = self.input_projection(x) # Conv 1x1
        x = F.relu(x)

        '''Processing of the conditioning audio (24KhZ)'''
        if audio_conditioning.dim() == 2:  # Aggiungi una dimensione per i canali solo se necessario
            audio_conditioning = audio_conditioning.unsqueeze(1)
        audio_conditioning = self.conditioning_projection(audio_conditioning) # torch.Size([1, 64, 24000])
        audio_conditioning = interpolate_audio_signal(audio_conditioning, scale_factor=2) # torch.Size([1, 64, 24000])
        audio_conditioning = self.conditioning_norm(audio_conditioning)

        diffusion_step = self.diffusion_embedding(diffusion_step).to(device) # torch.Size([1, 512]) (see class DiffusionEmbedding)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, audio_conditioning, diffusion_step)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x) # torch.Size([1, 64, 16000])
        x = F.relu(x) # torch.tanh(x)
        x = self.output_projection(x) # torch.Size([1, 1, 16000]) same shape of the input
      
        x = x.squeeze(1) # to match the dimensionality
        
        return x # WATCH: this is the prediction of the noise!
    
    @torch.no_grad()
    def sample(self, steps, conditioning, audio_length=48000):
        '''
        This is only used during inference and testing'''    
          
        device = next(self.parameters()).device

        x_t = torch.randn((1, 1, audio_length)).to(device).float()
        conditioning = conditioning.to(device).float()

        # One denoising step per time
        for t in reversed(range(steps)):
            diffusion_step = torch.tensor([t]).to(device).float()

            # Predict the noise for the current time step
            predicted_noise = self(x_t, diffusion_step, conditioning).float()
            # Get alpha from the noise schedule
            alpha_t = self.noise_schedule[t]

            # Remove the noise
            x_t = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        return x_t  # *This* is the denoised audio   


#################################################################################################################################################

# If you are curious about the intermediate shapes, or you simply want to debug the code, you can use the following:

debug_residual_block = False # set to true to debug the residual block
debug_full_model = False # set to true to debug the whole model

if debug_residual_block:
  residual_block = ResidualBlock(n_mels=80, residual_channels=64, dilation=2)

  batch_size = 1
  audio_length = 48000  # just an example

  # Fake input
  x = torch.randn(batch_size, 64, audio_length)

  # Fake conditioning
  audio_conditioning = torch.randn(batch_size, 64, audio_length)

  # Fake diffusion step
  diffusion_step = torch.randn(batch_size, 512)

  # Call the forward of the ResidualBlock
  residual_out, skip_out = residual_block(x, audio_conditioning, diffusion_step)


if debug_full_model:

  device = 'cuda'
  batch_size = 1
  audio_length = 48000  
  audio = torch.randn(batch_size, audio_length).to(device)
  audio_conditioning = torch.randn(1, 24000).to(device)
  diffusion_step = torch.tensor([10]).to(device)

  noise_schedule = np.linspace(1e-4, 0.05, 50)
  diffusion_steps = torch.randint(0, len(noise_schedule), (batch_size,), dtype=torch.long)

  # Define the parameters
  class Params:
      def __init__(self):
          self.residual_channels = 64
          self.noise_schedule = noise_schedule
          self.unconditional = False
          self.n_mels = 80
          self.audio_conditioning = torch.randn(batch_size, audio_length)
          self.residual_layers = 5
          self.dilation_cycle_length = 10

  params = Params()

  # Initialize the model
  model = DiffWave(params).to(device)

  total_params = count_parameters(model)
  print(f"Total number of parameters: {total_params}")

  model(audio, diffusion_step, audio_conditioning)