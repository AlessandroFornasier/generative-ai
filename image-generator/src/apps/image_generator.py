from os import name
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pick import pick
from pathlib import Path

parent_path = Path(__file__).resolve().parent.parent.parent
src_path = parent_path / 'src'
sys.path.append(str(src_path))

from architectures.unet import UNet
from data.mnist import MNISTDataLoader
from solvers.eueler import EulerSolver, SolverSolution
from helpers.trainer import Trainer, FlowMatchingLoss
from path.path import ProbabilityPath

def train_and_save_model(
    trainer : Trainer, 
    dataloader : DataLoader, 
    writer : SummaryWriter, 
    models_path : str, 
    model_name : str
  ) -> None:
    trainer.train(dataloader, models_path, model_name)
    trainer.save(models_path, model_name)
    epoch_files = glob.glob(f'{models_path}/{model_name}_epoch_*.pt')
    for file in epoch_files:
        Path(file).unlink()
    writer.flush()


def load_model(path: str, name: str, model: nn.Module) -> None:
    """
    Load a model.

    Args:
      path (str): Path relative to the model folder where the model weights are saved
      name (str): name of the model weights
    """
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(f'{path}/{name}.pt'))
    model.to(device)
    print(f"Model: {name} loaded from {path}/{name}.pt")


def generate(model: nn.Module, dataloader: DataLoader, path: str, n: int = 5) -> None:
    """
    """
    device = next(model.parameters()).device
    sample_batch = next(iter(dataloader))
    sample_shape = sample_batch[0].shape[1:]
    x0 = torch.randn(1, *sample_shape, device=device)
    time_grid = torch.linspace(0, 1, steps=100, dtype=torch.float32, device=device)
    euler_solver = EulerSolver(model, step_size=0.01, time_grid=time_grid)
    solution = euler_solver.solve(x0)
    fig, ax = plt.subplots(1, n, figsize=(n, n))
    for i in range(n):
        idx = len(solution.x) // n * i
        sample = solution.x[idx].squeeze().cpu().numpy()
        ax[i].imshow(sample, cmap='gray')
        ax[i].axis('off')
    plt.savefig(f'{path}_samples.png')
  
  
if __name__ == '__main__':
    models_path = f'{str(parent_path)}/models'
    runs_path = f'{str(parent_path)}/runs'
    generated_path = f'{str(parent_path)}/generated'
    data_path = f'{str(parent_path)}/data/MNIST'

    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(runs_path).mkdir(parents=True, exist_ok=True)
    Path(generated_path).mkdir(parents=True, exist_ok=True)
    Path(data_path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f'Model_{timestamp}'

    dataloader = MNISTDataLoader(batch_size=128).get_dataloader(data_path=data_path, train=True)

    learning_rate = 5e-5
    epochs = 20

    model = UNet(dims=[28, 28], n=1, k=448, m=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    path = ProbabilityPath()
    loss = FlowMatchingLoss()

    title = 'Select one among the following option:'
    options = ['generate', 'train', 'finetune']
    selected = pick(options, title)[0]

    model_names = [f.stem for f in Path(models_path).glob('*.pt') if f.is_file()]

    if selected == 'generate':
        title = 'Select the model to use for generation:'
        selected = pick(model_names, title)[0] 
        load_model(models_path, selected, model)
        generate(model, dataloader, f'{generated_path}/{selected}')
    elif selected == 'train':
        writer = SummaryWriter(f'{runs_path}/{model_name}')
        trainer = Trainer(device=device, model=model, path=path, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)
        train_and_save_model(trainer, dataloader, writer, models_path, model_name)
        writer.close()
    elif selected == 'finetune':
        title = 'Select the model to finetune:'
        selected = pick(model_names, title)[0]
        load_model(models_path, selected, model) 
        writer = SummaryWriter(f'{runs_path}/{model_name}')
        trainer = Trainer(device=device, model=model, path=path, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)
        train_and_save_model(trainer, dataloader, writer, models_path, model_name)
        writer.close()
    else:
        print("Invalid option selected.")