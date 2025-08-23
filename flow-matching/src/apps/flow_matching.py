from os import name
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pick import pick
from pathlib import Path

parent_path = Path(__file__).resolve().parent.parent.parent
src_path = parent_path / 'src'
sys.path.append(str(src_path))

from architectures.mlp import FlowMatchingMLP
from solvers.eueler import EulerSolver, SolverSolution
from helpers.trainer import Trainer, FlowMatchingLoss
from path.path import ProbabilityPath
from data.dataloader import MultimodalGaussianDataloader, LettersDatasetDataloader

def train_and_save_model(
    trainer : Trainer, 
    dataloader : DataLoader, 
    writer : SummaryWriter, 
    models_path : str, 
    model_name : str
  ) -> None:
    trainer.train(dataloader)
    trainer.save(models_path, model_name)
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


def generate_path(model: nn.Module) -> SolverSolution:
    """
    Generate a path using the Euler solver.
    
    Args:
        model (nn.Module): The flow matching model.
    
    Returns:
        SolverSolution: The solution containing the generated path and time steps.
    """
    device = next(model.parameters()).device
    x0 = torch.randn((1000, 2), dtype=torch.float32, device=device)
    time_grid = torch.linspace(0, 1, steps=100, dtype=torch.float32, device=device)
    euler_solver = EulerSolver(model, step_size=0.01, time_grid=time_grid)
    return euler_solver.solve(x0)


def visualize_data(dataloader: DataLoader, num_samples: int, filename: str) -> None:
    """
    Visualizes a batch of data from the dataloader.

    Args:
        dataloader (DataLoader): The dataloader containing the data.
        num_samples (int): Number of samples to visualize.
    """
    data_iter = iter(dataloader)
    points_collected = 0
    all_points = []
    
    while points_collected < num_samples:
        try:
            data = next(data_iter)
            remaining = num_samples - points_collected
            points_to_take = min(data.size(0), remaining)
            all_points.append(data[:points_to_take])
            points_collected += points_to_take
        except StopIteration:
            break
    
    plt.figure(figsize=(20, 20))
    
    if all_points:
        points = torch.cat(all_points, dim=0).cpu().detach().numpy()
        plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.tight_layout()
    # plt.show()

    plt.savefig(f'{filename}.png')
    print(f"Visualized {num_samples} samples from the dataloader and saved to {filename}.png.")


def visualize_path(solution: SolverSolution, filename: str):
    """
    Visualizes the probability path of the flow matching path.

    Args:
        solutions (torch.Tensor): Solutions from the Euler solver.
    """
    _, axs = plt.subplots(1, 10, figsize=(20, 2))
    
    indices = torch.linspace(0, len(solution.t) - 1, steps=10).long()

    for i, idx in enumerate(indices):
        points = solution.x[idx].cpu().detach().numpy()
        axs[i].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
        axs[i].set_title(f't={solution.t[idx].item():.2f}')
        axs[i].set_aspect('equal')
        axs[i].set_xlim(-10, 10)
        axs[i].set_ylim(-10, 10)

    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{filename}.png")
    print(f"Probability path visualized and saved to {filename}.png")


if __name__ == '__main__':
    # models_path = f'{str(parent_path)}/models/MGD'
    # runs_path = f'{str(parent_path)}/runs/MGD'
    # generated_path = f'{str(parent_path)}/generated/MGD'
    
    models_path = f'{str(parent_path)}/models/Letters'
    runs_path = f'{str(parent_path)}/runs/Letters'
    generated_path = f'{str(parent_path)}/generated/Letters'

    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(runs_path).mkdir(parents=True, exist_ok=True)
    Path(generated_path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # model_name = f'MGD_{timestamp}'
    model_name = f'Letters_{timestamp}'

    samples = 1000000
    # modes = 6
    # dataloader_factory = MultimodalGaussianDataloader(n_samples=samples, n_modes=modes, dim=2, scale=7.0, std=0.05, seed=1, batch_size=1000, shuffle=True)
    dataloader_factory = LettersDatasetDataloader(n_samples=samples, seed=1, std=0.25, batch_size=1000, shuffle=True)

    dims = [2, 512, 512, 512, 512, 2]
    learning_rate = 1e-3
    epochs = 20

    model_factory = FlowMatchingMLP(dims)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloader = dataloader_factory()
    model = model_factory()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    path = ProbabilityPath()
    loss = FlowMatchingLoss()

    title = 'Select one among the following option:'
    options = ['generate', 'train', 'visualize dataset']
    selected = pick(options, title)[0]

    model_names = [f.stem for f in Path(models_path).glob('*.pt') if f.is_file()]

    if selected == 'generate':
        title = 'Select the model to use for generation:'
        selected = pick(model_names, title)[0] 
        load_model(models_path, selected, model)
        visualize_path(generate_path(model), f'{generated_path}/{selected}_path')
    elif selected == 'visualize dataset':
        samples = 100000
        # visualize_data(dataloader, samples , f'{generated_path}/MGD_{samples}_sample_dataset')
        visualize_data(dataloader, samples , f'{generated_path}/Letters_{samples}_sample_dataset')
    else:
        writer = SummaryWriter(f'{runs_path}/{model_name}')
        trainer = Trainer(device=device, model=model, path=path, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)
        train_and_save_model(trainer, dataloader, writer, models_path, model_name)
        model_names.append(model_name)
        writer.close()