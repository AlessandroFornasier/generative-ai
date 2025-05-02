import os
import sys

import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import List, OrderedDict

parent_path = Path(__file__).resolve().parent.parent.parent
src_path = parent_path / 'src'
sys.path.append(str(src_path))

from architectures.vae import VAE, VAELoss
from data.mnist_loader import MNISTLoader
from helpers.trainer import Trainer

def vae_encoder_decoder(dims: List[int], binary: bool):
  """
  Defines the VAE encoder and decoder structure.

  Args:
    dims (List[int]): List of dimesions of the autoencoder layers
    binary (bool): Flag to indicate whether the autoencoder processes binary data [0, 1]

  Note:
    The output of the final encoder layer has double the size for mean and variance and no activation.
    If binary the activation of the last decoder layer is a Sigmoid to normalize to (0, 1)
  """
  encoder = []
  for n, (idim, odim) in enumerate(zip(dims[:-2], dims[1:-1])):
    encoder.append((f'Linear_{n}', nn.Linear(idim, odim)))
    encoder.append((f'SiLU_{n}', nn.SiLU()))
  encoder.append((f'Linear_{len(dims) - 2}', nn.Linear(dims[-2], 2 * dims[-1])))

  decoder = []
  for n, (idim, odim) in enumerate(zip(dims[-1:1:-1], dims[-2:0:-1])):
    decoder.append((f'Linear_{n}', nn.Linear(idim, odim)))
    decoder.append((f'SiLU_{n}', nn.SiLU()))
  decoder.append((f'Linear_{len(dims) - 2}', nn.Linear(dims[1], dims[0])))

  if binary:
    decoder.append((f'Sigmoid_{len(dims) - 2}', nn.Sigmoid()))

  return encoder, decoder

  
if __name__ == '__main__':
  binary = True
  dims = [28*28, 512, 256, 128, 64, 24, 12, 6, 3, 2]
  learning_rate = 1e-3
  weight_decay = 1e-2
  epochs = 50

  data_path = f'{str(parent_path)}/data/MNIST'
  models_path = f'{str(parent_path)}/models/MNIST'
  runs_path = f'{str(parent_path)}/runs/MNIST'

  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  model_name = f'VAE_{timestamp}'

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  encoder, decoder = vae_encoder_decoder(dims=dims, binary=binary)
  model = VAE(encoder=OrderedDict(encoder), decoder=OrderedDict(decoder))
  loss = VAELoss(binary=binary)
  dataloader = MNISTLoader(batch_size=128)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  writer = SummaryWriter(f'{runs_path}/{model_name}')
  trainer = Trainer(device=device, model=model, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)
  
  trainer.train_and_save(dataloader.get_dataloader(data_path=data_path, train=True), models_path, model_name)
  writer.flush()

  model_names = [f for f in models_path.glob('*.pt') if f.is_file()]
  model_names.append(model_name)

  for model_name in model_names:
    trainer.load_and_test(dataloader.get_dataloader(train=False), models_path, model_name)
  writer.flush()

  writer.close()

