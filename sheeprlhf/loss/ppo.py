from typing import Optional

import torch
import torch.nn.functional as F


def policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coeff: float,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the policy loss for PPO."""
    log_ratio = (log_probs - old_log_probs) * action_mask
    ratio = torch.exp(log_ratio)
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff)
    policy_loss = torch.max(policy_loss_1, policy_loss_2)
    if action_mask is not None:
        policy_loss = torch.sum(policy_loss * action_mask) / action_mask.sum()
    else:
        policy_loss = policy_loss.mean()
    return policy_loss


def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coeff: float,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the value loss for PPO."""
    values_clipped = torch.clamp(values, old_values - clip_coeff, old_values + clip_coeff)
    value_loss1 = F.mse_loss(values, returns, reduction="none")
    value_loss2 = F.mse_loss(values_clipped, returns, reduction="none")
    value_loss = torch.max(value_loss1, value_loss2)
    if action_mask is not None:
        value_loss = torch.sum(value_loss * action_mask) / action_mask.sum()
    else:
        value_loss = value_loss.mean()
    return value_loss
