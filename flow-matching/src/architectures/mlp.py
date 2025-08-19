import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) class.

    Args:
        layers (OrderedDict[str, nn.Module]): Ordered dictionary of MLP layers.
    """

    def __init__(self, layers: OrderedDict[str, nn.Module]) -> None:
      super(MLP, self).__init__()
      
      self.network = nn.Sequential(*layers.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Forward pass of the MLP.

      Args:
        x (torch.Tensor): Input tensor.

      Returns:
        torch.Tensor: Output tensor after passing through the MLP.
      """
      return self.network(x)
    

class FlowMatchingMLP:
  """
  Factory class for the MLP for flow matching.

  Args:
    dims (List[int]): List of dimesions of the MLP layers

  Note:
    The input layer has an extra dimension for the time embedding.
  """
  def __init__(self, dims: List[int]):
    self.dims = dims

  def __call__(self):
    layers = []
    layers.append(('Linear_0', nn.Linear(self.dims[0] + 1, self.dims[1])))
    layers.append(('SiLU_0', nn.SiLU()))
    for n, (idim, odim) in enumerate(zip(self.dims[1:-2], self.dims[2:-1]), 1):
      layers.append((f'Linear_{n}', nn.Linear(idim, odim)))
      layers.append((f'SiLU_{n}', nn.SiLU()))
    layers.append((f'Linear_{len(self.dims) - 2}', nn.Linear(self.dims[-2], self.dims[-1])))

    return MLP(OrderedDict(layers))