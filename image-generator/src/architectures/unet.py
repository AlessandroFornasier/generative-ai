import torch
import torch.nn as nn
import math
import pdb

from typing import List, Union
from itertools import pairwise


class Embedding(nn.Module):
  """
  Embedding.

  Args:
    dim (int): Dimension of the embedding.
  """
  def __init__(self, dim: int) -> None:
    super(Embedding, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(1, dim),
      nn.SiLU(),
      nn.Linear(dim, dim),
      nn.SiLU()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the Embedding.

    Args:
      x (torch.Tensor): Input tensor (batch_Size, 1).

    Returns:
      torch.Tensor: Output tensor after passing through the Embedding (batch_Size, m).
    """
    return self.network(x)


class ResidualBlock(nn.Module):
  """
  Fixed channels residual block with pre activation and time embedding.
  
  Args:
   - i (int): Number of input image's channels.
   - o (int): Number of output image's channels.
   - m (int): Dimension of the time and guidance embedding.
   - kernel_size (int): Size of the convolutional kernel.
   - stride (int): Stride of the convolution.
   - padding (int): Padding for the convolution.
  """

  @staticmethod
  def _groups(channels: int, groups: int) -> int:
      return math.gcd(channels, min(channels, groups)) or 1
    
  def __init__(self, i: int, o: int, m: int, kernel_size: int = 3, groups: int = 32) -> None:
    super(ResidualBlock, self).__init__()

    self.conv1 = nn.Conv2d(i, o, kernel_size=kernel_size, stride=1, padding=1)
    self.norm1 = nn.GroupNorm(self._groups(i, groups), i)
    self.conv2 = nn.Conv2d(o, o, kernel_size=kernel_size, stride=1, padding=1)
    self.norm2 = nn.GroupNorm(self._groups(o, groups), o)
    self.act = nn.SiLU()
    self.res = nn.Identity() if i == o else nn.Conv2d(i, o, kernel_size=1, stride=1)
    self.linear = nn.Linear(2 * m, o)

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the pre activated residual block.
    
    Args:
     - x (torch.Tensor): Input tensor of shape (batch_Size, n, height, width).
     - t (torch.Tensor): Embedded time tensor of shape (batch_Size, m).
     - y (torch.Tensor): Guidance tensor of shape (batch_Size, m).

    Returns:
     - torch.Tensor: Output tensor of shape (batch_Size, n, height, width) after passing through the residual block.
    """
    res = self.res(x)
    x = self.norm1(x)
    x = self.act(x)
    x = self.conv1(x)
    l = self.linear(torch.cat((t, y), dim=-1))
    l = l.unsqueeze(-1).unsqueeze(-1)
    x = self.norm2(x + l)
    x = self.act(x)
    x = self.conv2(x)
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
  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for module in self:
      if isinstance(module, ResidualBlock):
        x = module(x, t, y)
      else:
        x = module(x)
    return x


class UNet(nn.Module):
  """
  UNet.
  
  Args:
   - n (int): Number of input image's channels.
   - dims (List[int]): Number of latent space channels. Defines the depth of the UNet.
   - m (int): Dimension of the time embedding.
   - kernel_size (int): Size of the convolutional kernel.
   - groups (int): Number of groups for the GroupNorm.
  """
  def __init__(self, n: int, dims: List[int], m: int, kernel_size: int = 3, groups: int = 32) -> None:
    super(UNet, self).__init__()

    self.encoders = nn.ModuleList([])
    self.decoders = nn.ModuleList([])

    self.time_embedding = Embedding(m)
    self.guidance_embedding = Embedding(m)
    
    # Initial convolution: (batch_Size, n, height, width) -> (batch_Size, n, height, width)
    self.initial = nn.Conv2d(n, n, kernel_size=kernel_size, stride=1, padding=1)

    # Encoders: (batch_Size, n, height, width) -> (batch_Size, k, height / 2 * len(dims), width / 2 * len(dims))
    for id, od in pairwise([n] + dims):
      self.encoders.append(
        SequentialWrapper(
          ResidualBlock(id, id, m, kernel_size, groups),
          ResidualBlock(id, od, m, kernel_size, groups),
          Downsample(od, kernel_size)
        )
      )

    # Bottleneck: (batch_Size, dims[-1], height / 2 * len(dims), width / 2 * len(dims)) -> (batch_Size, dims[-1], height / 2 * len(dims), width / 2 * len(dims))
    self.bottleneck = SequentialWrapper(
      ResidualBlock(dims[-1], dims[-1], m, kernel_size, groups),
      ResidualBlock(dims[-1], dims[-1], m, kernel_size, groups),
      ResidualBlock(dims[-1], dims[-1], m, kernel_size, groups)
    )
    
    # Decoders: (batch_Size, dims[-1] * 2, height / 2 * len(dims), width / 2 * len(dims)) -> (batch_Size, n, height, width)
    self.decoders = nn.ModuleList([])
    for id, od in pairwise(dims[::-1] + [n]):
      self.decoders.append(
        SequentialWrapper(
          Upsample(id, kernel_size),
          ResidualBlock(id * 2, id, m, kernel_size, groups),
          ResidualBlock(id, od, m, kernel_size, groups)
        )
      )

    # Final convolution: (batch_Size, n, height, width) -> (batch_Size, n, height, width)  
    self.final = nn.Conv2d(n, n, kernel_size=kernel_size, stride=1, padding=1)

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the UNet.
    
    Args:
      x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
      t (torch.Tensor): Time tensor of shape (batch_size, time_dim).
      y (torch.Tensor): Guidance tensor of shape (batch_size, guidance_dim).

    Returns:
      torch.Tensor: Output tensor after passing through the encoder.
    """
    t = self.time_embedding(t)
    y = self.guidance_embedding(y)
    x = self.initial(x)
    
    skip_connections = []
    for layers in self.encoders:
      first, second, downsample = layers
      x = first(x, t, y)
      x = second(x, t, y)
      skip_connections.append(x)
      x = downsample(x)

    x = self.bottleneck(x, t, y)

    for layers in self.decoders:
      upsample, first, second = layers
      x = upsample(x)
      x = torch.cat((x, skip_connections.pop()), dim=1)
      x = first(x, t, y)
      x = second(x, t, y)

    x = self.final(x)
    return x