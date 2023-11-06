import torch


def finetune_loss(
    outputs: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100, label_smoothing: float = 0.0
) -> torch.Tensor:
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=ignore_index, label_smoothing=label_smoothing
    )
    return loss
