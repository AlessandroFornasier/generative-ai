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
from data.dataloader import MultimodalGaussianDataset, MoonsDataset

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
    _, axs = plt.subplots(2, 5, figsize=(10, 4))

    indices = torch.linspace(0, len(solution.t) - 1, steps=10).long()

    axs_dims = []
    for i, idx in enumerate(indices):
        points = solution.x[idx].cpu().detach().numpy()
        axs[i // 5, i % 5].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
        axs[i // 5, i % 5].set_title(f't={solution.t[idx].item():.2f}')
        axs[i // 5, i % 5].set_aspect('equal')
        axs_dims.append((axs[i // 5, i % 5].get_xlim(), axs[i // 5, i % 5].get_ylim()))

    x_lim = max([max(lim[0]) for lim in axs_dims])
    y_lim = max([max(lim[1]) for lim in axs_dims])
    
    for i in range(10):
        axs[i // 5, i % 5].set_xlim(-x_lim, x_lim)
        axs[i // 5, i % 5].set_ylim(-y_lim, y_lim)

    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{filename}.png")
    print(f"Probability path visualized and saved to {filename}.png")


if __name__ == '__main__':
    
    title = 'Select one among the following option:'
    options = ['generate', 'train', 'visualize dataset']
    action = pick(options, title)[0]

    title = 'Select one of the following supported distributions:'
    options = ['2D Gaussian Mixture', 'Moons']
    dataset = pick(options, title)[0]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    if dataset == '2D Gaussian Mixture':
        models_path = f'{str(parent_path)}/models/MGD'
        runs_path = f'{str(parent_path)}/runs/MGD'
        generated_path = f'{str(parent_path)}/generated/MGD'
        model_name = f'MGD_{timestamp}'
    elif dataset == 'Moons':
        models_path = f'{str(parent_path)}/models/Moons'
        runs_path = f'{str(parent_path)}/runs/Moons'
        generated_path = f'{str(parent_path)}/generated/Moons'
        model_name = f'Moons_{timestamp}'


    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(runs_path).mkdir(parents=True, exist_ok=True)
    Path(generated_path).mkdir(parents=True, exist_ok=True)

    if action != 'generate':
        if dataset == '2D Gaussian Mixture':
            print('Enter number of modes:')
            modes = int(input())
            dataloader = DataLoader(MultimodalGaussianDataset(n_samples=10000, n_modes=modes, dim=2, scale=7.0, std=0.05, seed=1), batch_size=1000, shuffle=True)
        elif dataset == 'Moons':
            dataloader = DataLoader(MoonsDataset(n_samples=10000, noise=0.1, seed=1), batch_size=1000, shuffle=True)

    dims = [2, 512, 512, 512, 512, 2]
    learning_rate = 1e-3

    model_factory = FlowMatchingMLP(dims)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model_factory()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    path = ProbabilityPath()
    loss = FlowMatchingLoss()

    model_names = [f.stem for f in Path(models_path).glob('*.pt') if f.is_file()]

    if action == 'generate':
        title = 'Select the model to use for generation:'
        selected = pick(model_names, title)[0] 
        load_model(models_path, selected, model)
        visualize_path(generate_path(model), f'{generated_path}/{selected}_path')
    elif action == 'visualize dataset':
        if dataset == '2D Gaussian Mixture':
            visualize_data(dataloader, 10000 , f'{generated_path}/MGD{modes}_dataset')
        elif dataset == 'Moons':
            visualize_data(dataloader, 10000 , f'{generated_path}/Moons_dataset')
    else:
        print('Enter number of epochs:')
        epochs = int(input())
        writer = SummaryWriter(f'{runs_path}/{model_name}')
        trainer = Trainer(device=device, model=model, path=path, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)
        train_and_save_model(trainer, dataloader, writer, models_path, model_name)
        model_names.append(model_name)
        writer.close()