from typing import TYPE_CHECKING, Dict, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sheeprlhf.agent import DPOAgent


def compute_masked_logprobs(
    logprobs: torch.Tensor, targets: torch.Tensor, ignore_index: int, average: bool = False
) -> torch.Tensor:
    """Retrieve log probabilities for targets, ignoring padding."""
    targets = targets[:, 1:].clone()
    loss_mask = targets != ignore_index
    if average:
        return (logprobs * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logprobs * loss_mask).sum(-1)


def dpo_loss(
    batch: Dict[str, torch.Tensor],
    agent: DPOAgent,
    beta: float,
    ignore_index: int,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45C1-L50C110."""
    chosen_input_ids = batch["chosen_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    chosen_targets = batch["chosen_targets"]
    rejected_input_ids = batch["rejected_input_ids"]
    rejected_attention_mask = batch["rejected_attention_mask"]
    rejected_targets = batch["rejected_targets"]

    with torch.inference_mode():
        ref_chosen_logprobs = agent.reference(
            input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False
        )
        ref_rejected_logprobs = agent.reference(
            input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False
        )
    actor_chosen_logprobs = agent.actor(
        input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False
    )
    actor_rejected_logprobs = agent.actor(
        input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False
    )

    masked_actor_chosen_logps = compute_masked_logprobs(actor_chosen_logprobs, chosen_targets, ignore_index)
    masked_actor_rejected_logps = compute_masked_logprobs(actor_rejected_logprobs, rejected_targets, ignore_index)
    masked_reference_chosen_logps = compute_masked_logprobs(ref_chosen_logprobs, chosen_targets, ignore_index)
    masked_reference_rejected_logps = compute_masked_logprobs(ref_rejected_logprobs, rejected_targets, ignore_index)

    actor_logratios = masked_actor_chosen_logps - masked_actor_rejected_logps
    ref_logratios = masked_reference_chosen_logps - masked_reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = actor_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (masked_actor_chosen_logps - masked_reference_chosen_logps).detach()
    rejected_rewards = beta * (masked_actor_rejected_logps - masked_reference_rejected_logps).detach()
    return losses.mean(), chosen_rewards, rejected_rewards
