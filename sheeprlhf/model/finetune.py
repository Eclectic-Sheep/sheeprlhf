from typing import Optional

import torch

from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.utils.lora import add_lora


class FinetuneModel(torch.nn.Module):
    """Base class for adapting finetuning for different models."""

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg

    def setup_finetuning(self, model_cfg: Optional[ModelConfig] = None) -> None:
        """Finetuning setup for parameters."""
        if model_cfg is None:
            model_cfg = self.model_cfg
        finetune_mode = self.model_cfg.finetune_mode
        if finetune_mode == FINETUNE_MODE.ALL:
            for param in self.parameters():
                param.requires_grad = True

        elif finetune_mode == FINETUNE_MODE.LORA:
            for param in self.parameters():
                param.requires_grad = False
            add_lora(self, lora_cfg=self.model_cfg.lora_cfg)

        else:
            raise ValueError(f"Unknown finetuning mode {finetune_mode}")
