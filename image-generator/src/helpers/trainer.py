import os
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Callable

from path.path import ProbabilityPath

class FlowMatchingLoss:
    """
    Conditional Flow Matching Loss.
    
    This class implements the Conditional Flow Matching loss function.
    """

    def __call__(self, u: torch.Tensor, u_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Conditional Flow Matching loss.

        Args:
            u (torch.Tensor): Conditional vector field ut(x|z).
            u_target (torch.Tensor): Target conditional vector field ut(x|z).

        Returns:
            loss (torch.Tensor): Computed loss value.
        """
        loss = torch.mean((u - u_target) ** 2)
        return loss
    

class Trainer:
  """
  Trainer class for training a flow matching model.

  Args:
    device (torch.device): The device (CPU or GPU) to run the model on.
    model (nn.Module): The model to be trained.
    path (ProbabilityPath): The probability path sampler.
    loss (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function for training.
    optimizer (Optimizer): The optimizer for training the model.
    epochs (int): Number of training epochs.
    writer (Optional[SummaryWriter]): Optional TensorBoard writer for logging training metrics.
  """
  def __init__(
    self,
    device: torch.device,
    model: nn.Module,
    path: ProbabilityPath,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    epochs: int,
    writer: Optional[SummaryWriter] = None,
  ) -> None:
    self.device = device
    self.model = model.to(self.device)
    self.loss = loss
    self.path = path
    self.optimizer = optimizer
    self.epochs = epochs
    self.writer = writer
    self.models_folder = './models'

  def train(self, dataloader: DataLoader, models_path: Optional[str] = None, model_name: Optional[str] = None) -> None:
    """
    Trains the flow matching model on the given dataset.

    Args:
      dataloader (DataLoader): The DataLoader for loading training data.
      models_path (Optional[str]): Path to save model checkpoints. If None, checkpoints are not saved.
      model_name (Optional[str]): Name of the model for saving checkpoints. Required if models_path
    """
    self.model.train()

    step = 0
    for epoch in range(self.epochs):
      epoch_loss = 0.0
      progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

      for batch_idx, (data, _) in enumerate(progress_bar):
        z = data.to(self.device)
        t = torch.rand(data.size(0), 1, device=self.device)
        sample = self.path.sample(t, z)
        u = self.model(sample.x, t)
        loss = self.loss(u, sample.u)

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

        step += 1
        tqdm.write(f"Epoch [{epoch+1}/{self.epochs}] - Batch loss: {batch_loss:.4f} - Epoch Loss: {epoch_loss:.4f} - Avg Loss: {average_loss:.4f}")
        
      if models_path and model_name:
        self.save(models_path, f"{model_name}_epoch_{epoch+1}")
        if epoch > 0:
          old_checkpoint = f"{models_path}/{model_name}_epoch_{epoch}.pt"
          if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)
        

  def save(self, path: str, name: str):
    """
    Save the model weight.

    Args:
      path (str): Path to save the model weights
      name (str): name of the model weights
    """
    os.makedirs(path, exist_ok=True)
    torch.save(self.model.state_dict(), f'{path}/{name}.pt')
    print(f'Model {name} saved to: {path}/{name}')

  def load(self, path: str, name: str):
    """
    Load a model.

    Args:
      path (str): Path relative to the model folder where the model weights are saved
      name (str): name of the model weights
    """
    self.model.load_state_dict(torch.load(f'{path}/{name}.pt'))
    print(f"Model: {name} loaded from {path}/{name}.pt")
