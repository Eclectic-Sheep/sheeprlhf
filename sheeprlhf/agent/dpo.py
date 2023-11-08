from typing import Optional

from lightning import Fabric

from sheeprlhf.model.actor import ActorModel
from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.utils.logger import trainable_parameter_summary
from sheeprlhf.utils.lora import add_lora, disable_lora, enable_lora
from sheeprlhf.utils.model import get_model_checkpoint


class DPOAgent:
    """Agent model for DPO training."""

    _reference: ActorModel
    _finetune_mode: FINETUNE_MODE
    _actor: Optional[ActorModel] = None
    _lora_enabled: bool
    _sft_checkpoint_path: str
    _sft_model_cfg: ModelConfig

    def __init__(
        self,
        fabric: Fabric,
        model_cfg: ModelConfig,
        sft_experiment_dir: str,
        sft_model_name: Optional[str] = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.fabric = fabric
        # Currently we only support same architecture for reference and actor models
        self._sft_model_cfg, self._sft_checkpoint_path = get_model_checkpoint(sft_experiment_dir, sft_model_name)
        self._lora_enabled = model_cfg.finetune_mode == FINETUNE_MODE.LORA

        self._reference = ActorModel(model_cfg=self._sft_model_cfg)
        self._finetune_mode = model_cfg.finetune_mode

        if not self._lora_enabled:
            self._actor = ActorModel(model_cfg=self._sft_model_cfg)

    def load_checkpoint(self) -> None:
        """Load checkpoint for both actor and reference model."""
        self._reference.load_checkpoint(
            path=self._sft_checkpoint_path, fabric=self.fabric, model_cfg=self._sft_model_cfg, freeze=True
        )
        if not self._lora_enabled:
            self._actor.load_checkpoint(
                path=self._sft_checkpoint_path, fabric=self.fabric, model_cfg=self._sft_model_cfg, freeze=False
            )

    def setup_finetuning(self, model_cfg: Optional[ModelConfig] = None) -> None:
        """Finetuning setup for both actor and reference model."""
        if model_cfg is not None:
            model_cfg = self.model_cfg
        self.fabric.print("Loading actor model")
        if self.lora_enabled:
            self.fabric.print("Adding LORA parameters to reference model")
            add_lora(self._reference, lora_cfg=self.model_cfg.lora_cfg)
            trainable_parameter_summary(self._reference, show_names=False, fabric=self.fabric)
        else:
            self._actor.setup_finetuning(model_cfg)
            trainable_parameter_summary(model=self._actor, show_names=False, fabric=self.fabric)

    @property
    def actor(self) -> ActorModel:  # noqa: D102
        if self._finetune_mode == FINETUNE_MODE.LORA:
            enable_lora(self._reference)
            return self._reference
        else:
            return self._actor

    @property
    def reference(self) -> ActorModel:  # noqa: D102
        if self._finetune_mode == FINETUNE_MODE.LORA:
            disable_lora(self._reference)
        return self._reference
