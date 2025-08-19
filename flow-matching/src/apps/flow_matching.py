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
from data.dataloader import MultimodalGaussianDataloader


def visualize_path(solution: SolverSolution):
    """
    Visualizes the path of the flow matching path.

    Args:
        solutions (torch.Tensor): Solutions from the Euler solver.
    """
    fig, axs = plt.subplots(1, 10, figsize=(20,20))
    
    indices = torch.linspace(0, len(solution.t) - 1, steps=10).long()

    for i, idx in enumerate(indices):
        points = solution.x[idx].cpu().detach().numpy()
        axs[i].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
        axs[i].set_title(f't={solution.t[idx].item():.2f}')
        axs[i].set_aspect('equal')
        axs[i].set_xlim(-4, 4)
        axs[i].set_ylim(-4, 4)
        
    plt.tight_layout()
    plt.savefig("flow_matching_path.png")
    

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


if __name__ == '__main__':
    models_path = f'{str(parent_path)}/models/MGD'
    runs_path = f'{str(parent_path)}/runs/MGD'
    generated_path = f'{str(parent_path)}/generated/MGD'

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f'MGD_{timestamp}'

    mgd_samples = 1000000
    mgd_modes = 6
    dataloader_factory = MultimodalGaussianDataloader(n_samples=mgd_samples, n_modes=mgd_modes, dim=2, std=0.05, seed=1, batch_size=1000, shuffle=True)

    dims = [2, 512, 512, 512, 512, 2]
    learning_rate = 1e-3
    epochs = 10

    model_factory = FlowMatchingMLP(dims)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = dataloader_factory()
    model = model_factory()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    path = ProbabilityPath()
    loss = FlowMatchingLoss()

    title = 'Select one among the following option:'
    options = ['generate', 'train']
    selected = pick(options, title)[0]

    model_names = [f.stem for f in Path(models_path).glob('*.pt') if f.is_file()]

    if selected == 'generate':
        title = 'Select the model to use for generation:'
        selected = pick(model_names, title)[0]
        # TODO: Generate from initial sample 
        # x0 = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
        # time_grid = torch.linspace(0, 1, steps=100, dtype=torch.float32, device=device)
        # euler_solver = EulerSolver(model, step_size=0.05, time_grid=time_grid)
        # solution = euler_solver.solve(x0)
        # visualize_path(solution)
    else:
        writer = SummaryWriter(f'{runs_path}/{model_name}')
        trainer = Trainer(device=device, model=model, path=path, loss=loss, optimizer=optimizer, epochs=epochs, writer=writer)
        train_and_save_model(trainer, dataloader, writer, models_path, model_name)
        model_names.append(model_name)
 
    writer.close()