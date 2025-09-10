import torch
import torch.nn as nn

from dataclasses import dataclass


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
    def __init__(self, model: nn.Module, step_size: float, time_grid: torch.Tensor):
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
            return self.model(x, t.unsqueeze(-1).repeat(x.shape[0], 1))
           
        self.model.eval()
            
        x = x0.detach().clone()
        solutions = [x.detach().clone()]
        times = [self.time_grid[0].detach().clone()]

        for t in self.time_grid[1:]:
            x = x + self.step_size * u(x, t)
            solutions.append(x.detach().clone())
            times.append(t.detach().clone())

        return SolverSolution(x=torch.stack(solutions), t=torch.stack(times))