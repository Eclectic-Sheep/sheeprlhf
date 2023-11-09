import os

import lightning
import torch
from lightning_utilities.core.rank_zero import rank_zero_only

from sheeprlhf.model.finetune import FinetuneModel
from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.utils.lora import add_lora, get_lora_state_dict, merge_lora
from sheeprlhf.utils.model import load_hf_transformer


class CasualModel(FinetuneModel):
    """Casual model for SFT training and casual generation."""

    def __init__(self, model_cfg: ModelConfig):
        super().__init__(model_cfg=model_cfg)
        self.model = load_hf_transformer(self.model_cfg)

    def forward(self, **kwargs):  # noqa: D102
        if self.training and not self.model_cfg.use_attention_mask:
            kwargs.pop("attention_mask")
        return self.model(**kwargs).logits

    def generate(self, **kwargs):  # noqa: D102
        return self.model.generate(**kwargs)

    def load_checkpoint(
        self,
        path: str,
        fabric: lightning.Fabric,
        model_cfg: ModelConfig,
        freeze: bool = False,
    ):
        """Loads a checkpoint from given path."""
        sd = torch.load(path, map_location=fabric.device)
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("model.", "")
            new_sd[new_k] = v
        if model_cfg.finetune_mode == FINETUNE_MODE.LORA:
            add_lora(self.model, lora_cfg=model_cfg.lora_cfg)
            self.model.load_state_dict(new_sd, strict=False)
            merge_lora(self.model)
        elif model_cfg.finetune_mode == FINETUNE_MODE.ALL:
            self.model.load_state_dict(new_sd)
        else:
            raise ValueError(f"Unknown finetune mode {model_cfg.finetune_mode}")
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    @rank_zero_only
    def save_checkpoint(self, fabric: lightning.Fabric, experiment_dir: str, model_cfg: ModelConfig, step):
        """Checkpoint saving for critic model.

        This includes saving the critic head model as well.
        """
        output_file = os.path.join(experiment_dir, "model", f"checkpoint-{step}.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sd = get_lora_state_dict(self) if model_cfg.finetune_mode == FINETUNE_MODE.LORA else self.state_dict()
        fabric.save(output_file, sd)
