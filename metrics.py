import numpy as np
import torch
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import accuracy_score

__all__ = [
    "cox_log_rank",
    "CIndex_lifeline",
    "MultiLabel_Acc",
]


def cox_log_rank(hazards: torch.Tensor,
                 labels: torch.Tensor,
                 survtime: torch.Tensor) -> float:
    """
    Compute log-rank p-value after dichotomising the risk scores.
    """
    hazards_np = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazards_np)
    risk_bin = (hazards_np > median).astype(int)

    survtime = survtime.cpu().numpy().reshape(-1)
    labels = labels.cpu().numpy()

    idx_low = risk_bin == 0
    T_low, T_high = survtime[idx_low], survtime[~idx_low]
    E_low, E_high = labels[idx_low], labels[~idx_low]

    p_value = logrank_test(
        T_low, T_high,
        event_observed_A=E_low,
        event_observed_B=E_high
    ).p_value
    return float(p_value)


def CIndex_lifeline(hazards: torch.Tensor,
                    labels: torch.Tensor,
                    survtime: torch.Tensor) -> float:
    """
    Concordance index (higher is better, 0.5 is random).
    """
    return concordance_index(
        survtime.cpu().numpy(),
        -hazards.cpu().numpy().reshape(-1),
        labels.cpu().numpy()
    )


def MultiLabel_Acc(pred: torch.Tensor,
                   target: torch.Tensor) -> np.ndarray:
    """
    Column-wise accuracy for multi-label classification.
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    acc = np.array([
        accuracy_score(target[:, i], pred[:, i])
        for i in range(target.shape[1])
    ])
    return acc
