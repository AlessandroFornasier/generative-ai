import torch
import torch.nn as nn

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoencoderState:
  """
  Autoencoder state.

  Attributes:
   - x (Optional[torch.Tensor]): Input data
   - z (Optional[torch.Tensor]): Latent space sample
   - x_hat (Optional[torch.Tensor]): Reconstructed data
  """
  x : Optional[torch.Tensor] = None                         # Input
  z : Optional[torch.Tensor] = None                         # Latent space sample
  x_hat : Optional[torch.Tensor] = None                     # Reconstructed input


class Autoencoder(nn.Module):
  """
  Autoencoder class.

  Args:
    encoder (OrderedDict[str, nn.Module]): Ordered dictionary of encoder layers
    decoder (OrderedDict[str, nn.Module]): Ordered dictionary of decoder layers
  """

  def __init__(
    self,
    encoder: OrderedDict[str, nn.Module],
    decoder: OrderedDict[str, nn.Module]
  ) -> None:
    super(Autoencoder, self).__init__()

    self.encoder = nn.Sequential(encoder)
    self.decoder = nn.Sequential(decoder)

    def encode(self, x) -> torch.Tensor:
      """
      Encodes the input data into the latent space.

      Args:
        x (torch.Tensor): Input data.

      Returns:
        z (torch.Tensor): Encoded data, latent space.
      """
      return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
      """
      Decodes the latent space data.

      Args:
        z (torch.Tensor): Latent space data.

      Returns:
        x_hat (torch.Tensor): Decoded data, output space.
      """
      return self.decoder(z)

    def forward(self, x) -> torch.Tensor:
      """
      Forward pass of the autoencoder
      """
      state = AutoencoderState(x)
      state.z = self.encode(x)
      state.x_hat = self.decode(state.z)
      return state
