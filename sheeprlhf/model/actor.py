import typing

import torch.nn.functional as F

from sheeprlhf.model.casual import CasualModel

if typing.TYPE_CHECKING:
    from sheeprlhf.structure.model import ModelConfig


class ActorModel(CasualModel):
    """Actor model for PPO and DPO algorithms."""

    def __init__(self, model_cfg: ModelConfig):
        super().__init__(model_cfg=model_cfg)

    def forward(self, **kwargs):  # noqa: D102
        input_ids = kwargs["input_ids"]
        if self.training and not self.model_cfg.use_attention_mask:
            kwargs.pop("attention_mask")
        out = self.model(**kwargs)
        # Model predicts next token log probability here.
        actor_log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        selected_actor_log_probs = actor_log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        return selected_actor_log_probs.squeeze(-1)
