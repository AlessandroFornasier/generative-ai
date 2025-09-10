import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional

@dataclass
class SchedulerState:
    """
    Conditional OT (Optimal Transport) scheduler state.

    Attributes:
    - alpha (Optional[torch.Tensor]): alpha
    - beta (Optional[torch.Tensor]): beta
    - alpha_dot (Optional[torch.Tensor]): time derivative of alpha
    - beta_dot (Optional[torch.Tensor]): time derivative of beta
    """
    alpha : Optional[torch.Tensor] = None
    beta : Optional[torch.Tensor] = None
    alpha_dot : Optional[torch.Tensor] = None
    beta_dot : Optional[torch.Tensor] = None


class Scheduler:
    """
    Conditional OT (Optimal Transport) scheduler
    """
    def __call__(self, t: torch.Tensor) -> SchedulerState:
        return SchedulerState(
            alpha = t,
            beta = 1 - t,
            alpha_dot = torch.ones_like(t),
            beta_dot = -torch.ones_like(t),
        )


@dataclass
class ProbabilityPathSample:
    """
    Sample from the conditional OT (Optimal Transport) (Gaussian) probability path.

    Attributes:
    - x (torch.Tensor): Sampled data from conditional probability path pt(x|z).
    - u (torch.Tensor): Conditional target vector field ut(x|z).
    """
    x: torch.Tensor
    u: torch.Tensor


class ProbabilityPath:
    """
    Simple conditional OT (Optimal Transport) (Gaussian) probability probability path
    """

    def __init__(self, scheduler: Optional[Scheduler] = None) -> None:
        if scheduler is None:
            scheduler = Scheduler()
        self.scheduler = scheduler

    def sample(self, t: torch.Tensor, z: torch.Tensor) -> ProbabilityPathSample:
        """
        Sample the conditional OT (Optimal Transport) (Gaussian) probability path pt(x|z) and 
        returns the conditional OT (Optimal Transport) (Gaussian) target vector field.

        Args:
            t (torch.Tensor): Time tensor.
            z (torch.Tensor): Data sample.

        Returns:
            x (torch.Tensor): Sampled data from conditional probability path pt(x|z).
            u (torch.Tensor): Conditional target vector field ut(x|z).
        """
        state = self.scheduler(t)
        alpha = state.alpha
        beta = state.beta
        
        while alpha.dim() < z.dim():
            alpha = alpha.unsqueeze(-1)
        while beta.dim() < z.dim():
            beta = beta.unsqueeze(-1)
        
        eps = torch.randn_like(z)
        x = alpha * z + beta * eps
        u = z - eps

        # In the general Gaussian case the vector field would be computed as:
        # u = (alpha_dot - beta_dot * alpha / beta) * z + beta_dot * x / beta

        return ProbabilityPathSample(x, u)