import torch
import torch.nn as nn

class Generator:
  """
  Sampler class for sampling data from latent space and generate data via decoding.

  Args:
    device (torch.device): The device (CPU or GPU) to run the model on.
    model (nn.Module): The autoencoder model to be trained.
    n (int): Size of the latent space
  """
  def __init__(
    self,
    device: torch.device,
    model: nn.Module,
    n: int
  ) -> None:
    self.device = device
    self.model = model.to(self.device)
    self.n = n

  def generate(self) -> torch.Tensor:
    """
    Sample from the latent space and generate data via decoding.
    """
    self.model.eval()
    z = torch.randn(self.n).to(self.device)
    with torch.no_grad():
      return self.model.decode(z)
    
  def load(self, path: str, name: str):
    """
    Load a model.

    Args:
      path (str): Path relative to the model folder where the model weights are saved
      name (str): name of the model weights
    """
    self.model.load_state_dict(torch.load(f'{path}/{name}.pt'))
    print(f"Model: {name} loaded from {path}/{name}.pt")


