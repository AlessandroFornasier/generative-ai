import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import List, OrderedDict
from pick import pick

parent_path = Path(__file__).resolve().parent.parent.parent
src_path = parent_path / 'src'
sys.path.append(str(src_path))

from architectures.vae import VAE, VAELoss
from data.mnist_loader import MNISTLoader
from helpers.trainer import Trainer
from helpers.generator import Generator
from mnist_vae import MNISTVAE


def train_and_save_model(
    trainer : Trainer, 
    dataloader : MNISTLoader, 
    writer : SummaryWriter, 
    data_path : str, 
    models_path : str, 
    model_name : str
  ) -> None:
    trainer.train(dataloader.get_dataloader(data_path=data_path, train=True))
    trainer.save(models_path, model_name)
    writer.flush()


def finetune_and_save_model(
    trainer : Trainer, 
    dataloader : MNISTLoader, 
    writer : SummaryWriter, 
    data_path : str, 
    models_path : str, 
    model_name : str
  ) -> None:
  trainer.load(models_path, model_name)
  trainer.train(dataloader.get_dataloader(data_path=data_path, train=True))
  trainer.save(models_path, model_name)
  writer.flush()



def test_models(
    trainer : Trainer, 
    dataloader : MNISTLoader, 
    writer : SummaryWriter, 
    data_path : str, 
    models_path : str, 
    model_names : List[str]
  ) -> None:
  for model_name in model_names:
    trainer.load(models_path, model_name)
    trainer.test(dataloader.get_dataloader(data_path=data_path, train=False))
  writer.flush()


def generate_sample(
    generator : Generator,
    models_path : str, 
    model_name : str,
    generated_path: str,
    n : int
  ) -> None:
  generator.load(models_path, model_name)
  fig, ax = plt.subplots(n, n, figsize=(n, n))
  for i in range(n):
    for j in range(n):
      sample = generator.generate()
      ax[i][j].imshow(sample.view(28, 28).cpu().detach().numpy(), cmap='gray')
      ax[i][j].axis('off')
  plt.savefig(f'{generated_path}/samples.png')

if __name__ == '__main__':
  binary = True
  dims = [28*28, 512, 256, 128, 64, 32, 12]
  learning_rate = 1e-3
  beta = 0.8
  epochs = 10

  data_path = f'{str(parent_path)}/data/MNIST'
  models_path = f'{str(parent_path)}/models/MNIST'
  runs_path = f'{str(parent_path)}/runs/MNIST'
  generated_path = f'{str(parent_path)}/generated/MNIST'

  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  model_name = f'VAE_{timestamp}'

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  encoder, decoder = MNISTVAE(dims=dims, binary=binary)()
  model = VAE(encoder=OrderedDict(encoder), decoder=OrderedDict(decoder))
  loss = VAELoss(binary=binary, beta=beta)
  dataloader = MNISTLoader(batch_size=128)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  title = 'Select one among the following option:'
  options = ['generate', 'test', 'train', 'finetune']
  selected = pick(options, title)[0]

  model_names = [f.stem for f in Path(models_path).glob('*.pt') if f.is_file()]

  if selected == 'generate':
    generator = Generator(device=device, model=model, n=dims[-1])
    title = 'Select the model to use for generation:'
    selected = pick(model_names, title)[0]
    generate_sample(generator, models_path, selected, generated_path, 10)
  else:
    writer = SummaryWriter(f'{runs_path}/{model_name}')
    trainer = Trainer(device=device, model=model, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)

    if selected == 'test':
      title = 'Select the model to test:'
      options = model_names + ['all']
      selected = pick(options, title)[0]
      if selected == 'all':
        test_models(trainer, dataloader, writer, data_path, models_path, model_names)
      else:
        test_models(trainer, dataloader, writer, data_path, models_path, [selected])    
    elif selected == 'train':  
      train_and_save_model(trainer, dataloader, writer, data_path, models_path, model_name)
      model_names.append(model_name)
    elif selected == 'finetune':
      writer = SummaryWriter(f'{runs_path}/{model_name}')
      title = 'Select the model to finetune:'
      options = model_names
      model_name = pick(options, title)[0]
      finetune_and_save_model(trainer, dataloader, writer, data_path, models_path, model_name)
      
    writer.close()


