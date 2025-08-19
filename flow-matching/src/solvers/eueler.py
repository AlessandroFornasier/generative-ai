import torch

from dataclasses import dataclass
from architectures.mlp import MLP


@dataclass
class SolverSolution:
    """
    State of the solver.

    Attributes:
    - x (torch.Tensor): solutions for each timestep.
    - t (torch.Tensor): time steps.
    """
    x: torch.Tensor
    t: torch.Tensor


class EulerSolver:
    def __init__(self, model: MLP, step_size: float, time_grid: torch.Tensor):
        self.model = model
        self.step_size = step_size
        self.time_grid = time_grid

    def solve(self, x0: torch.Tensor) -> SolverSolution:
        """
        Solve the ODE using the Euler method.

        Args:
            x0 (torch.Tensor): Initial condition.

        Returns:
            SolverState: State containing solution and time steps.
        """
        def u(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.model(torch.cat([x, t.unsqueeze(-1).repeat(x.shape[0], 1)], dim=-1))
            
        x = x0.clone()
        solutions = [x.clone()]
        times = [self.time_grid[0].clone()]

        for t in self.time_grid[1:]:
            x = x + self.step_size * u(x, t)
            solutions.append(x.clone())
            times.append(t.clone())

        return SolverSolution(x=torch.stack(solutions), t=torch.stack(times))