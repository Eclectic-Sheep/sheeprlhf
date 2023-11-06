import lightning
import torch

from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.utils.lora import add_lora


class FinetuneModel(torch.nn.Module):
    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg

    def setup_finetuning(self, fabric: lightning.Fabric):
        finetune_mode = self.model_cfg.finetune_mode
        if finetune_mode == FINETUNE_MODE.ALL:
            fabric.print("Using all layers parameters for finetuning")
            for param in self.parameters():
                param.requires_grad = True

        elif finetune_mode == FINETUNE_MODE.LORA:
            fabric.print("Adding LORA parameters for finetuning")
            for param in self.parameters():
                param.requires_grad = False
            add_lora(self, lora_cfg=self.model_cfg.lora_cfg)

        else:
            raise ValueError(f"Unknown finetuning mode {finetune_mode}")
