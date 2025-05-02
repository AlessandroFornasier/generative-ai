import torch
import torch.nn as nn

class Generator:
  """
  Sampler class for sampling data from latent space and generate data via decoding.

  Args:
    device (torch.device): The device (CPU or GPU) to run the model on.
    model (nn.Module): The autoencoder model to be trained.
  """
  def __init__(
    self,
    device: torch.device,
    model: nn.Module,
  ) -> None:
    self.device = device
    self.model = model.to(self.device)

  def generate(self,  n: int) -> torch.Tensor:
    """
    Sample from the latent space and generate data via decoding.
    """
    self.model.eval()
    z = torch.randn(n)
    with torch.no_grad():
      return self.model.decode(z)

