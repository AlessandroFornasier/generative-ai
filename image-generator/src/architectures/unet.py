import torch
import torch.nn as nn
import math
import pdb

from typing import List, Union


class TimeEmbedding(nn.Module):
  """
  Time embedding.

  Args:
    dim (int): Dimension of the time embedding.
  """
  def __init__(self, dim: int) -> None:
    super(TimeEmbedding, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(1, dim),
      nn.SiLU(),
      nn.Linear(dim, dim),
      nn.SiLU()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the TimeEmbedding.

    Args:
      x (torch.Tensor): Input tensor (batch_Size, 1).

    Returns:
      torch.Tensor: Output tensor after passing through the TimeEmbedding (batch_Size, m).
    """
    return self.network(x)
  

class ResidualBlock(nn.Module):
  """
  Fixed channels residual block with pre activation and time embedding.
  
  Args:
   - n (int): Number of image's channels.
   - m (int): Dimension of the time embedding.
   - kernel_size (int): Size of the convolutional kernel.
   - stride (int): Stride of the convolution.
   - padding (int): Padding for the convolution.
  """

  def __init__(self, n: int, m: int, kernel_size: int = 3, groups: int = 32) -> None:
    super(ResidualBlock, self).__init__()
    
    self.conv = nn.Conv2d(n, n, kernel_size=kernel_size, stride=1, padding=1)
    self.norm = nn.GroupNorm(groups, n)
    self.res = nn.Identity()
    self.linear_time = nn.Linear(m, n)

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the pre activated residual block.
    
    Args:
     - x (torch.Tensor): Input tensor of shape (batch_Size, n, height, width).
     - t (torch.Tensor): Embedded time tensor of shape (batch_Size, m).
      
    Returns:
     - torch.Tensor: Output tensor of shape (batch_Size, n, height, width) after passing through the residual block.
    """
    res = self.res(x)
    x = self.norm(x)
    x = nn.SiLU()(x)
    x = self.conv(x)
    t = self.linear_time(t)
    t = t.unsqueeze(-1).unsqueeze(-1)
    x = self.norm(x + t)
    x = nn.SiLU()(x)
    x = self.conv(x)
    return x + res


class Downsample(nn.Module):
  """
  Downsampling block.
  
  Args:
   - n (int): Number of image's channels.
   - kernel_size (int): Size of the convolutional kernel.
  """
  def __init__(self, n: int, kernel_size: int = 3, padding: Union[int, tuple] = 1) -> None:
    super(Downsample, self).__init__()
    self.conv = nn.Conv2d(n, n, kernel_size=kernel_size, stride=2, padding=padding)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the downsampling block.
    
    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
      
    Returns:
      torch.Tensor: Output tensor after downsampling.
    """
    return self.conv(x)
  
  
class Upsample(nn.Module):
  """
  Upsampling block.
  
  Args:
   - n (int): Number of image's channels.
   - kernel_size (int): Size of the convolutional kernel.
  """
  def __init__(self, n: int, kernel_size: int = 3, padding: Union[int, tuple] = 1) -> None:
    super(Upsample, self).__init__()
    self.conv = nn.ConvTranspose2d(n, n, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the upsampling block.
    
    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
      
    Returns:
      torch.Tensor: Output tensor after upsampling.
    """
    return self.conv(x)


class SequentialWrapper(nn.Sequential):
  """
  A wrapper for nn.Sequential for passing the tensor x and the time embedding t.
  """
  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    for module in self:
      if isinstance(module, ResidualBlock):
        x = module(x, t)
      else:
        x = module(x)
    return x


class UNet(nn.Module):
  """
  UNet.
  
  Args:
   - dims (List[int]): image dimensions (height, width).
   - n (int): Number of image's channels.
   - k (int): Number of latent space channels.
   - m (int): Dimension of the time embedding.
   - kernel_size (int): Size of the convolutional kernel.
   - groups (int): Number of groups for the GroupNorm.
  """
  def __init__(self, dims: List[int], n: int, k: int, m: int, kernel_size: int = 3, groups: int = 32) -> None:
    super(UNet, self).__init__()

    self.time_embedding = TimeEmbedding(m)

    padding = tuple(1 + (2 ** math.ceil(math.log2(dim)) - dim) // 2 for dim in dims)
    kernel = tuple(kernel_size + 2 ** math.ceil(math.log2(dim)) - dim for dim in dims)

    self.encoders = nn.ModuleList([
      # Initial convolution + downsample: (batch_Size, n, height, width) -> (batch_Size, k, height / 2, width / 2)
      SequentialWrapper(
        nn.Conv2d(n, k, kernel_size=kernel_size, stride=1, padding=padding),
        Downsample(k, kernel_size)
      ),
      # Encoder: (batch_Size, k, height / 2, width / 2) -> (batch_Size, k, height / 4, width / 4)
      SequentialWrapper(
        ResidualBlock(k, m, kernel_size, groups),
        ResidualBlock(k, m, kernel_size, groups),
        Downsample(k, kernel_size)
      ),
      # # Encoder: (batch_Size, k, height / 4, width / 4) -> (batch_Size, k, height / 8, width / 8)
      # SequentialWrapper(
      #   ResidualBlock(k, m, kernel_size, groups),
      #   ResidualBlock(k, m, kernel_size, groups),
      #   Downsample(k, kernel_size)
      # ),
      # # Encoder: (batch_Size, k, height / 8, width / 8) -> (batch_Size, k, height / 16, width / 16)
      # SequentialWrapper(
      #   ResidualBlock(k, m, kernel_size, groups),
      #   ResidualBlock(k, m, kernel_size, groups),
      #   Downsample(k, kernel_size)
      # )
    ])
  
    # Bottleneck: (batch_Size, k, height / 16, width / 16) -> (batch_Size, k, height / 16, width / 16)  
    self.bottleneck = SequentialWrapper(
      ResidualBlock(k, m, kernel_size, groups),
      ResidualBlock(k, m, kernel_size, groups),
      ResidualBlock(k, m, kernel_size, groups)
    )
    
    self.decoders = nn.ModuleList([
      # Decoder: (batch_Size, 2k, height / 16, width / 16) -> (batch_Size, 2k, height / 8, width / 8)
      SequentialWrapper(
        Upsample(2 * k, kernel_size),
        ResidualBlock(2 * k, m, kernel_size, groups),
        ResidualBlock(2 * k, m, kernel_size, groups)
      ),
      # # Decoder: (batch_Size, 3k, height / 8, width / 8) -> (batch_Size, 3k, height / 4, width / 4)
      # SequentialWrapper(
      #   Upsample(3 * k, kernel_size),
      #   ResidualBlock(3 * k, m, kernel_size, groups),
      #   ResidualBlock(3 * k, m, kernel_size, groups)
      # ),
      # # Decoder: (batch_Size, 4k, height / 4, width / 4) -> (batch_Size, 4k, height / 2, width / 2)
      # SequentialWrapper(
      #   Upsample(4 * k, kernel_size),
      #   ResidualBlock(4 * k, m, kernel_size, groups),
      #   ResidualBlock(4 * k, m, kernel_size, groups)
      # ),
      # # Final Upsample + convolution: (batch_Size, 5k, height / 2, width / 2) -> (batch_Size, n, height, width)
      # SequentialWrapper(
      #   Upsample(5 * k, kernel_size),
      #   nn.Conv2d(5 * k, n, kernel_size=kernel , stride=1, padding=1)
      # )
      SequentialWrapper(
        Upsample(3 * k, kernel_size),
        nn.Conv2d(3 * k, n, kernel_size=kernel , stride=1, padding=1)
      )
    ])
    
  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the UNet.
    
    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
      t (torch.Tensor): Time tensor of shape (batch_size, time_dim).
      
    Returns:
      torch.Tensor: Output tensor after passing through the encoder.
    """
    t = self.time_embedding(t)
    
    skip_connections = []
    for layers in self.encoders:
      x = layers(x, t)
      skip_connections.append(x)
      
    x = self.bottleneck(x, t)

    for layers in self.decoders:
      x = torch.cat((x, skip_connections.pop()), dim=1)
      x = layers(x, t)
      
    return x