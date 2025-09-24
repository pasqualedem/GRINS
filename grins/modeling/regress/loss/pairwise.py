import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskWeightedPairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking hinge loss with optional automatic task weighting (Kendall et al.).

    Params:
        num_tasks: int
            Total number of tasks (size of model output).
        use_task_weighting: bool
            If True, learnable uncertainty weights per task are applied.
    """

    def __init__(
        self, num_tasks: int, margin: float = 1.0, use_task_weighting: bool = False
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.margin = margin
        self.use_task_weighting = use_task_weighting

        if use_task_weighting:
            # Learnable sigmas (one per task)
            sigmas = torch.ones(num_tasks, requires_grad=True)
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.sigmas = None

    def forward(self, scores0, scores1, task_idxs, labels):
        """
        Args:
            scores0, scores1: [B, num_tasks] model outputs
            task_idxs: [B] indices of the active task for each sample
            labels: [B] ∈ {0,1}
        Returns:
            Scalar loss
        """
        # Convert labels {0,1} → {+1, -1}
        y = -(
            2 * labels.to(scores0.device).float() - 1.0
        )  # [B], +1 means scores0 should be higher, -1 means scores1 should be higher

        # Compute per-sample hinge loss
        per_sample_loss = torch.clamp(
            self.margin
            - y
            * (
                scores0.gather(1, task_idxs.unsqueeze(-1)).squeeze(-1)
                - scores1.gather(1, task_idxs.unsqueeze(-1)).squeeze(-1)
            ),
            min=0.0,
        )  # [B]

        loss_per_task = []
        if self.use_task_weighting:
            # Compute loss **per task**
            for t in range(self.num_tasks):
                mask = task_idxs == t
                if mask.any():
                    task_loss = per_sample_loss[mask].mean()
                    sigma = self.sigmas[t]
                    # Kendall uncertainty weighting
                    weighted_loss = 0.5 / (sigma**2) * task_loss + torch.log(
                        1 + sigma**2
                    )
                    loss_per_task.append(weighted_loss)
            # Sum over tasks
            total_loss = torch.stack(loss_per_task).sum()
        else:
            # Just mean over all samples
            total_loss = per_sample_loss.mean()

        return total_loss, loss_per_task


class TaskWeightedPairwiseLogisticRankingLoss(nn.Module):
    """
    Pairwise logistic ranking loss with optional automatic task weighting (Kendall et al.).

    Params:
        num_tasks: int
            Total number of tasks (size of model output).
        use_task_weighting: bool
            If True, learnable uncertainty weights per task are applied.
    """

    def __init__(self, num_tasks: int, use_task_weighting: bool = False):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_task_weighting = use_task_weighting

        if use_task_weighting:
            # Learnable sigmas (one per task)
            sigmas = torch.ones(num_tasks, requires_grad=True)
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.sigmas = None

    def forward(self, scores0, scores1, task_idxs, labels):
        """
        Args:
            scores0, scores1: [B, num_tasks] model outputs
            task_idxs: [B] indices of the active task for each sample
            labels: [B] ∈ {0,1}
        Returns:
            Scalar loss, list of per-task losses
        """
        # Convert labels {0,1} → {+1, -1}
        y = -(2 * labels.to(scores0.device).float() - 1.0)  # [B]

        # Extract active task scores
        s0 = scores0.gather(1, task_idxs.unsqueeze(-1)).squeeze(-1)  # [B]
        s1 = scores1.gather(1, task_idxs.unsqueeze(-1)).squeeze(-1)  # [B]

        # Pairwise logistic loss per sample
        per_sample_loss = F.softplus(-y * (s0 - s1))  # log(1+exp(-margin))

        loss_per_task = []
        if self.use_task_weighting:
            # Compute loss **per task**
            for t in range(self.num_tasks):
                mask = task_idxs == t
                if mask.any():
                    task_loss = per_sample_loss[mask].mean()
                    sigma = self.sigmas[t]
                    # Kendall uncertainty weighting
                    weighted_loss = 0.5 / (sigma**2) * task_loss + torch.log(
                        1 + sigma**2
                    )
                    loss_per_task.append(weighted_loss)
            # Sum over tasks
            total_loss = torch.stack(loss_per_task).sum()
        else:
            # Just mean over all samples
            total_loss = per_sample_loss.mean()

        return total_loss, loss_per_task


class TaskPairwiseCorrect(nn.Module):
    """
    Computes pairwise correct predictions per batch for multi-task ranking.
    For each sample, only the score of the active task is considered.
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores0, scores1, task_idxs, labels):
        """
        Args:
            scores0, scores1: [B, num_tasks] model outputs
            task_idxs: [B] indices of the active task for each sample
            labels: [B] ∈ {0,1}  (0 → scores0 should win, 1 → scores1 should win)
        Returns:
            accuracy: scalar tensor
        """
        # Select scores of the relevant task
        s0 = scores0.gather(1, task_idxs.unsqueeze(-1)).squeeze(-1)  # [B]
        s1 = scores1.gather(1, task_idxs.unsqueeze(-1)).squeeze(-1)  # [B]

        # Labels: 0 means s0 should be higher, 1 means s1 should be higher
        pred = (s0 < s1).long()  # 1 if s1 wins, 0 if s0 wins
        correct = (pred == labels.long()).float()  # [B]

        return correct