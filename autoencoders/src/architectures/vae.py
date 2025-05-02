import torch
import torch.nn as nn
import torch.nn.functional as Func

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict
from architectures.autoencoder import Autoencoder, AutoencoderState

@dataclass
class VAEState(AutoencoderState):
  """
  Variational autoencoder state.

  Attributes:
   - x (Optional[torch.Tensor]): Input data
   - z (Optional[torch.Tensor]): Latent space sample
   - x_hat (Optional[torch.Tensor]): Reconstructed data
   - dist (Optional[torch.distributions.Distribution]): Encoder Gaussian distribution
  """
  dist : Optional[torch.distributions.Distribution] = None  # Latent space distribution


class VAE(Autoencoder):
  """
  Variational autoencoder class.

  Args:
    encoder (OrderedDict[str, nn.Module]): Ordered dictionary of encoder layers
    decoder (OrderedDict[str, nn.Module]): Ordered dictionary of decoder layers
  """

  def __init__(
    self,
    encoder: OrderedDict[str, nn.Module],
    decoder: OrderedDict[str, nn.Module]
  ) -> None:
    super().__init__(encoder, decoder)

    self.softplus = nn.Softplus()

  def encode(self, x, eps: float = 1e-6) -> torch.distributions.Distribution:
    """
    Encodes the input data into the latent space.

    Args:
      x (torch.Tensor): Input data.
      eps (float): Small value to avoid numerical instability.

    Returns:
      dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.

    Note:
      Learning logvar improves numerical stability since var is smaller than zero and tipically smaller than once. Hence logvar is within (-inf, log(1)).
      Softplus + epsilon (softplus(x) = \log(1 + \exp(x))) is used to get sigma instead of directly exponentiating while ensuring numerical stability
    """
    e = self.encoder(x)
    mu, logvar = torch.tensor_split(e, 2, dim=-1)
    var = self.softplus(logvar) + eps
    return torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag_embed(var)) # Use scale_tril as it is more efficient

  def reparametrize(self, dist) -> torch.Tensor:
    """
    Perform sampling via the reparametrization trick

    Args:
      dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.

    Returns:
      z (torch.Tensor): Sampled data from the latent space z = mu + sigma * epsilon. With epsilon ~ N(0,I)
    """
    return dist.rsample()

  def decode(self, z) -> torch.Tensor:
    """
    Decodes the data from the latent space to the original input space.

    Args:
      z (torch.Tensor): Data in the latent space.

    Returns:
      x_hat (torch.Tensor): Reconstructed data in the original input space.
    """
    return self.decoder(z)

  def forward(self, x) -> VAEState:
    """
    Performs a forward pass of the VAE.

    Args:
      x (torch.Tensor): Input data.

    Returns:
      state (VAEState): state of the VAE.
    """
    state = VAEState(x)
    state.dist = self.encode(state.x)
    state.z = self.reparametrize(state.dist)
    state.x_hat = self.decode(state.z)
    return state


class VAELoss:
  """
  VAE Loss callable class. The VAE loss is given by the ELBO,
  which is the the sum of the reconstruction loss and the KL divergence loss

  Note:
    A VAE is trained by maximizing ELBO:
    - Reconstruction loss (MSE ~ cross entropy)
    - KL divergence

  Refernce:
    - https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
    - https://github.com/pytorch/examples/blob/main/vae/main.py
  """
  def __init__(self, binary: bool) -> None:
    self.binary = binary

  def __binary_vae_loss(self, state: VAEState) -> Dict[str, torch.Tensor]:
    rl = Func.binary_cross_entropy(state.x_hat, state.x, reduction='none').sum(-1).mean() # Reconstruction loss
    target_dist = torch.distributions.MultivariateNormal(
      torch.zeros_like(state.z, device=state.z.device),
      scale_tril=torch.eye(state.z.shape[-1], device=state.z.device).unsqueeze(0).expand(state.z.shape[0], -1, -1),
    )
    kll = torch.distributions.kl.kl_divergence(state.dist, target_dist).mean() # KL loss
    return {"Reconstruction": rl, "KL": kll}

  def __call__(self, state: VAEState) -> Dict[str, torch.Tensor]:
    if self.binary:
      return self.__binary_vae_loss(state)
    else:
      raise NotImplementedError