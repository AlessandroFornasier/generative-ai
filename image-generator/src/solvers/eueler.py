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
    def __init__(self, model: nn.Module, step_size: float, time_grid: torch.Tensor, guidance_scale: float):
        self.model = model
        self.step_size = step_size
        self.time_grid = time_grid
        self.guidance_scale = guidance_scale

    def solve(self, x0: torch.Tensor, y: torch.Tensor) -> SolverSolution:
        """
        Solve the ODE using the Euler method.

        Args:
            x0 (torch.Tensor): Initial condition.
            y (torch.Tensor): Labels for conditional generation.

        Returns:
            SolverState: State containing solution and time steps.
        """
        def u(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            u_free = self.model(x, t.unsqueeze(-1).repeat(x.shape[0], 1), torch.full_like(y, -1))
            if self.guidance_scale == 1.0:
                return u_free
            u_guided = self.model(x, t.unsqueeze(-1).repeat(x.shape[0], 1), y)
            return (1 - self.guidance_scale) * u_free + self.guidance_scale * u_guided

        self.model.eval()
            
        x = x0.detach().clone()
        solutions = [x.detach().clone()]
        times = [self.time_grid[0].detach().clone()]

        for t in self.time_grid[1:]:
            x = x + self.step_size * u(x, t, y)
            solutions.append(x.detach().clone())
            times.append(t.detach().clone())

        return SolverSolution(x=torch.stack(solutions), t=torch.stack(times))