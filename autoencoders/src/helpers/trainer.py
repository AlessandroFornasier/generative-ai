import os
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Callable, Dict

from architectures.autoencoder import AutoencoderState


class Trainer:
  """
  Trainer class for training an autoencoder model.

  Args:
    device (torch.device): The device (CPU or GPU) to run the model on.
    model (nn.Module): The autoencoder model to be trained.
    loss (Callable[[AutoencoderState], Dict[torch.Tensor]]): A callable loss function that takes the model's state and returns a scalar loss value.
    optimizer (Optimizer): The optimizer for training the model.
    epochs (int): Number of epochs
    writer (Optional[SummaryWriter]): Optional TensorBoard writer for logging training metrics.
  """
  def __init__(
    self,
    device: torch.device,
    model: nn.Module,
    loss: Callable[[AutoencoderState], Dict[str, torch.Tensor]],
    optimizer: Optimizer,
    epochs: int,
    writer: Optional[SummaryWriter] = None,
  ) -> None:
    self.device = device
    self.model = model.to(self.device)
    self.loss = loss
    self.optimizer = optimizer
    self.epochs = epochs
    self.writer = writer
    self.models_folder = './models'

  def train(self, dataloader: DataLoader) -> None:
    """
    Trains the autoencoder model on the given dataset.

    Args:
      dataloader (DataLoader): The DataLoader for loading training data.
    """
    self.model.train()

    step = 0
    for epoch in range(self.epochs):
      epoch_loss = 0.0
      progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

      for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(self.device)

        state = self.model(data)
        losses = self.loss(state)

        loss = sum(losses.values())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        average_loss = epoch_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=batch_loss)

        if self.writer:
          self.writer.add_scalar("Train/Loss/Batch", batch_loss, step)
          self.writer.add_scalar("Train/Loss/Epoch", average_loss, step)
          for name, loss in losses.items():
            self.writer.add_scalar(f"Train/Loss/{name}", loss, step)

        step += 1
        print(f"Epoch [{epoch+1}/{self.epochs}] - Batch loss: {batch_loss:.4f} - Epoch Loss: {epoch_loss:.4f} - Avg Loss: {average_loss:.4f}")

  def train_and_save(self, dataloader: DataLoader, path: str, name: str):
    """
    Trains the autoencoder model on the given dataset and save the model weight.

    Args:
      dataloader (DataLoader): The DataLoader for loading training data.
      path (str): Path to save the model weights
      name (str): name of the model weights
    """
    self.train(dataloader=dataloader)
    os.makedirs(path, exist_ok=True)
    torch.save(self.model.state_dict(), f'{path}/{name}')
    print(f'Model {name} saved to: {path}/{name}')

  def test(self, dataloader: DataLoader) -> None:
    """
    Test the autoencoder model on the given dataset.

    Args:
      dataloader (DataLoader): The DataLoader for loading training data.
    """
    self.model.eval()

    average_loss = 0.0
    with torch.no_grad():
      for data, _ in tqdm(dataloader, desc="Testing"):
        data = data.to(self.device)

        state = self.model(data)
        losses = self.loss(state)
        loss = sum(losses.values())

        average_loss += loss.item()

    average_loss /= len(dataloader)
    if self.writer:
      self.writer.add_scalar("Test/Loss/Average", average_loss)
      for name, loss in losses.items():
        self.writer.add_scalar(f"Test/Loss/Average/{name}", loss / len(dataloader))

    print(f"Average test loss: {average_loss:.4f}")

  def load_and_test(self, dataloader: DataLoader, path: str, name: str):
    """
    Test the autoencoder model on the given dataset.

    Args:
      dataloader (DataLoader): The DataLoader for loading training data.
      path (str): Path relative to the model folder where the model weights are saved
      name (str): name of the model weights
    """
    self.model.load_state_dict(torch.load(f'{path}/{name}'))
    print(f"Model: {name}")
    self.test(dataloader=dataloader)
