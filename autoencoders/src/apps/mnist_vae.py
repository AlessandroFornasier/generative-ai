import torch
import torch.nn as nn

from typing import List

class MNISTVAE:
  """
  Defines the VAE encoder and decoder structure.

  Args:
    dims (List[int]): List of dimesions of the autoencoder layers
    binary (bool): Flag to indicate whether the autoencoder processes binary data [0, 1]

  Note:
    The output of the final encoder layer has double the size for mean and variance and no activation.
    If binary the activation of the last decoder layer is a Sigmoid to normalize to (0, 1)
  """
  def __init__(self, dims: List[int], binary: bool):
    self.dims = dims
    self.binary = binary

  def __call__(self):
    encoder = []
    for n, (idim, odim) in enumerate(zip(self.dims[:-2], self.dims[1:-1])):
      encoder.append((f'Linear_{n}', nn.Linear(idim, odim)))
      encoder.append((f'SiLU_{n}', nn.SiLU()))
    encoder.append((f'Linear_{len(self.dims) - 2}', nn.Linear(self.dims[-2], 2 * self.dims[-1])))

    decoder = []
    for n, (idim, odim) in enumerate(zip(self.dims[-1:1:-1], self.dims[-2:0:-1])):
      decoder.append((f'Linear_{n}', nn.Linear(idim, odim)))
      decoder.append((f'SiLU_{n}', nn.SiLU()))
    decoder.append((f'Linear_{len(self.dims) - 2}', nn.Linear(self.dims[1], self.dims[0])))

    if self.binary:
      decoder.append((f'Sigmoid_{len(self.dims) - 2}', nn.Sigmoid()))

    return encoder, decoder
