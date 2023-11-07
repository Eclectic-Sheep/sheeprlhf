from typing import Tuple

import torch
import torch.nn.functional as F

from sheeprlhf.structure.task import RM_LOSS_TYPE


def reward_loss_last_token(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Last token based reward loss computation.

    This loss computes the logsigmoid of the difference between the chosen and
    rejected rewards from last generated non-terminal token.

    Returns:
        loss: the mean loss for the batch
        chosen_last_rewards: the last reward for the chosen sequence for each example in the batch
        rejected_last_rewards: the last reward for the rejected sequence for each example in the batch
    """
    mask_chosen = chosen != pad_token_id
    mask_rejected = rejected != pad_token_id

    # last non-padding token is the terminal one
    # we want to retrieve reward that leads to that state(token)
    last_chosen_token_idx = torch.argmax(torch.cumsum(mask_chosen, dim=1) * mask_chosen, dim=1, keepdim=True) - 1
    last_rejected_token_idx = torch.argmax(torch.cumsum(mask_rejected, dim=1) * mask_rejected, dim=1, keepdim=True) - 1
    last_chosen_rewards = torch.gather(chosen_rewards, dim=-1, index=last_chosen_token_idx).squeeze(-1)
    last_rejected_rewards = torch.gather(rejected_rewards, dim=-1, index=last_rejected_token_idx).squeeze(-1)

    filtered_rewards = last_chosen_rewards - last_rejected_rewards
    return -F.logsigmoid(filtered_rewards).mean(), last_chosen_rewards, last_rejected_rewards


def reward_loss_average(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward loss computing the average of all tokens.

    This loss computes the logsigmoid of the difference between the chosen and
    rejected rewards from average of all output tokens excluding padding tokens.

    Returns:
        loss: the mean loss for the batch
        chosen_last_rewards: the last reward for the chosen sequence for each example in the batch
        rejected_last_rewards: the last reward for the rejected sequence for each example in the batch
    """
    mask_chosen = chosen != pad_token_id  # (B, T)
    mask_rejected = rejected != pad_token_id  # (B, T)

    divergence = ((chosen - rejected) != 0).int().argmax(1)

    # TODO: Can we implement it in vectorized way?
    for i, d in enumerate(divergence):
        mask_chosen[i, :d] = 0
        mask_rejected[i, :d] = 0

    # last non-padding token is the terminal one
    # we want to retrieve reward that leads to that state(token)
    # TODO: check if everytime this is true
    last_chosen_token_idx = torch.argmax(torch.cumsum(mask_chosen, dim=1) * mask_chosen, dim=1, keepdim=True) - 1
    last_rejected_token_idx = torch.argmax(torch.cumsum(mask_rejected, dim=1) * mask_rejected, dim=1, keepdim=True) - 1
    mask_chosen[:, last_chosen_token_idx + 1] = 0
    mask_rejected[:, last_rejected_token_idx + 1] = 0
    last_chosen_rewards = torch.gather(chosen_rewards, dim=-1, index=last_chosen_token_idx).squeeze(-1)
    last_rejected_rewards = torch.gather(rejected_rewards, dim=-1, index=last_rejected_token_idx).squeeze(-1)

    chosen_rewards_average = chosen_rewards * mask_chosen
    chosen_rewards_average = chosen_rewards_average.sum(dim=1) / mask_chosen.sum(dim=1)
    rejected_rewards_average = rejected_rewards * mask_rejected
    rejected_rewards_average = rejected_rewards_average.sum(dim=1) / mask_rejected.sum(dim=1)

    filtered_rewards = chosen_rewards_average - rejected_rewards_average
    return -F.logsigmoid(filtered_rewards).mean(), last_chosen_rewards, last_rejected_rewards


def reward_loss_per_sample(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per token reward loss.

    This loss computes the logsigmoid of the difference between the chosen and rejected rewards
    from every token in the sequence masked by the pad token id. It is adapted from
    https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py#L37
    for each example in the batch:
        - find the index where the chosen and rejected sequences diverge
        - find the index of the last non terminal token in the sequence
        - compute the loss for the tokens between the divergence index and the last token index

    Returns:
        loss: the mean loss for the batch
        chosen_last_rewards: the last reward for the chosen sequence for each example in the batch
        rejected_last_rewards: the last reward for the rejected sequence for each example in the batch
    """
    batch_size = chosen.shape[0]
    sequence_len = chosen.shape[1]
    loss: torch.Tensor = torch.tensor(0.0, device=chosen.device)
    chosen_last_rewards = []
    rejected_last_rewards = []
    total_num_samples = 0
    for i in range(batch_size):
        # Get the chosen and rejected actions for the current sample
        chosen_actions = chosen[i]
        rejected_actions = rejected[i]

        # Get the rewards for the chosen and rejected actions for the current example
        chosen_complete_rewards = chosen_rewards[i]
        rejected_complete_rewards = rejected_rewards[i]

        # Find the index where the action sequence diverge
        divergence_ind = (chosen_actions != rejected_actions).nonzero()
        divergence_ind = divergence_ind[0].item() if len(divergence_ind) > 0 else sequence_len - 1

        # Find padding tokens
        pad_mask_chosen = (chosen_actions == pad_token_id).nonzero()
        if chosen_actions[0] == pad_token_id:
            # If the first token is a pad token, we want to exclude it from the mask
            pad_mask_chosen = pad_mask_chosen[1:]
        chosen_last_index = pad_mask_chosen[0].item() if len(pad_mask_chosen) > 0 else chosen.shape[1]
        pad_mask_rejected = (rejected_actions == pad_token_id).nonzero()
        if rejected_actions[0] == pad_token_id:
            # If the first token is a pad token, we want to exclude it from the mask
            pad_mask_rejected = pad_mask_rejected[1:]
        rejected_last_index = pad_mask_rejected[0].item() if len(pad_mask_rejected) > 0 else rejected.shape[1]
        end_ind = max(chosen_last_index, rejected_last_index)

        if divergence_ind > end_ind:
            continue

        if divergence_ind == end_ind:
            # If the divergence index is the same as the end index, we want to include the last token
            divergence_ind -= 1

        # Get the rewards for the chosen and rejected sequences after the divergence index
        chosen_filtered_rewards = chosen_complete_rewards[divergence_ind:end_ind]
        rejected_filtered_rewards = rejected_complete_rewards[divergence_ind:end_ind]

        # Compute the loss for the current example
        filtered_rewards = chosen_filtered_rewards - rejected_filtered_rewards
        loss.add_(-F.logsigmoid(filtered_rewards).mean())

        # Get the last non-padding token rewards for the current example
        chosen_last_rewards.append(chosen_complete_rewards[chosen_last_index - 1])
        rejected_last_rewards.append(rejected_complete_rewards[rejected_last_index - 1])
        total_num_samples += 1

    loss.div_(total_num_samples)
    chosen_last_rewards = torch.stack(chosen_last_rewards)
    rejected_last_rewards = torch.stack(rejected_last_rewards)

    return loss, chosen_last_rewards, rejected_last_rewards


def load_reward_loss(reward_loss_type: str):
    """Helper function to select which type of reward loss to use."""
    if reward_loss_type == RM_LOSS_TYPE.AVERAGE:
        return reward_loss_average
    elif reward_loss_type == RM_LOSS_TYPE.LAST_TOKEN:
        return reward_loss_last_token
    elif reward_loss_type == RM_LOSS_TYPE.PER_SAMPLE:
        return reward_loss_per_sample
    else:
        raise ValueError(f"Invalid reward loss type: {reward_loss_type}")
