import torch
import torch.nn as nn

__all__ = [
    "surv_loss",
    "MultiTaskLossWrapper",
    "Identity",
]


def surv_loss(event: torch.Tensor,
              time: torch.Tensor,
              risk: torch.Tensor) -> torch.Tensor:
    """
    Negative log partial likelihood for the Cox model.
    """
    n = len(time)
    # R_ij = 1 if time_j >= time_i
    risk_matrix = (time.expand(n, n) <= time.expand(n, n).t()).float()
    theta = risk.reshape(-1)
    exp_theta = torch.exp(theta)
    loss = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * risk_matrix, dim=1))) *
        event.float()
    )
    return loss


class MultiTaskLossWrapper(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al.).
    """
    def __init__(self, task_num: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        assert len(losses) == len(self.log_vars), \
            "Number of provided losses must match task_num."
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total


class Identity(nn.Module):
    """A no-op layer used to replace the final classifier."""
    def forward(self, x):
        return x
