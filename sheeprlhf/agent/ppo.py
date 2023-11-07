from pathlib import Path
from typing import Optional

from lightning import Fabric
from omegaconf import OmegaConf

from sheeprlhf.model.actor import ActorModel
from sheeprlhf.model.critic import CriticModel
from sheeprlhf.model.reward import RewardModel
from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.utils.logger import trainable_parameter_summary
from sheeprlhf.utils.lora import add_lora, add_multiple_lora, disable_lora, enable_lora, select_lora
from sheeprlhf.utils.model import get_last_checkpoint_path


class PPOAgent:
    """Agent model for PPO training."""

    _reference: ActorModel
    _reward: RewardModel
    _finetune_mode: FINETUNE_MODE
    _actor: Optional[ActorModel] = None
    _critic: Optional[CriticModel] = None
    _same_actor_critic: bool = False
    _share_actor_critic: bool = False
    _share_critic_reward: bool = False

    _sft_checkpoint_path: str
    _sft_model_cfg: ModelConfig
    _rm_checkpoint_path: str
    _rm_model_cfg: ModelConfig

    _lora_enabled: bool
    _init_critic_with_rm: bool

    def __init__(
        self,
        fabric: Fabric,
        model_cfg: ModelConfig,
        init_critic_with_rm: bool,
        sft_experiment_dir: str,
        rm_experiment_dir: str,
    ) -> None:
        self.model_cfg = model_cfg
        self.fabric = fabric
        self._init_critic_with_rm = init_critic_with_rm

        sft_exp_cfg = OmegaConf.load(Path(sft_experiment_dir) / ".hydra/config.yaml")
        self._sft_model_cfg = ModelConfig(**sft_exp_cfg.model)
        self._sft_checkpoint_path = get_last_checkpoint_path(sft_experiment_dir)
        sft_model_name = self._sft_model_cfg.repo_name

        rm_exp_cfg = OmegaConf.load(Path(rm_experiment_dir) / ".hydra/config.yaml")
        self._rm_model_cfg = ModelConfig(**rm_exp_cfg.model)
        self._rm_checkpoint_path = get_last_checkpoint_path(rm_experiment_dir)
        rm_model_name = self._rm_model_cfg.repo_name

        self._reference = ActorModel(model_cfg=self._sft_model_cfg)
        self._reward = RewardModel(model_cfg=self._rm_model_cfg)

        self._same_actor_critic = sft_model_name == rm_model_name
        self._finetune_mode = model_cfg.finetune_mode
        self._lora_enabled = self._finetune_mode == FINETUNE_MODE.LORA

    def load_checkpoint(
        self,
    ) -> None:
        """Load checkpoints for Actor, Critic and Reward models."""
        self._reference.load_checkpoint(
            path=self._sft_checkpoint_path, fabric=self.fabric, model_cfg=self._sft_model_cfg, freeze=True
        )
        if not self._init_critic_with_rm:
            if not (self._lora_enabled and self._same_actor_critic):
                # Actor and critic cannot be shared, we fallback to the default behavior
                self._actor.load_checkpoint(
                    path=self._sft_checkpoint_path, fabric=self.fabric, model_cfg=self._sft_model_cfg, freeze=True
                )
                self._critic.load_checkpoint(
                    path=self._sft_checkpoint_path, fabric=self.fabric, model_cfg=self._sft_model_cfg, freeze=True
                )
        else:
            # here we have critic model initialized with reward model so we need separete actor model
            self._actor.load_checkpoint(
                path=self._sft_checkpoint_path, fabric=self.fabric, model_cfg=self._sft_model_cfg, freeze=True
            )
            if not self._lora_enabled:
                self._critic.load_checkpoint(
                    path=self._rm_checkpoint_path, fabric=self.fabric, model_cfg=self._rm_model_cfg, freeze=True
                )

    def setup_finetuning(self, model_cfg: Optional[ModelConfig] = None) -> None:
        """Setup finetuning for Actor, Critic and Reward models."""
        if model_cfg is None:
            model_cfg = self.model_cfg
        lora_cfg = self.model_cfg.lora_cfg
        if not self._init_critic_with_rm:
            if self._lora_enabled and self._same_actor_critic:
                # here we can share reference model between Actor and Critic
                add_multiple_lora(self._reference, lora_cfg=lora_cfg, num=2)
                trainable_parameter_summary(self._reference, show_names=False, fabric=self.fabric)
            else:
                # Actor and critic cannot be shared, we fallback to the default behavior
                self._actor.setup_finetuning(model_cfg=model_cfg)
                trainable_parameter_summary(self._actor, show_names=False, fabric=self.fabric)
                self._critic.setup_finetuning(model_cfg=model_cfg)
                trainable_parameter_summary(self._critic, show_names=False, fabric=self.fabric)
        else:
            # here we have critic model initialized with reward model so we need separete actor model
            self._actor.setup_finetuning(model_cfg=model_cfg)
            trainable_parameter_summary(self._actor, show_names=False, fabric=self.fabric)
            if self._lora_enabled:
                add_lora(self._reward, lora_cfg=lora_cfg)
                trainable_parameter_summary(self._reward, show_names=False, fabric=self.fabric)
            else:
                self._critic.setup_finetuning(model_cfg=model_cfg)
                trainable_parameter_summary(self._critic, show_names=False, fabric=self.fabric)

    @property
    def actor(self) -> ActorModel:  # noqa: D102
        if self._share_actor_critic:
            enable_lora(self._reference)
            return select_lora(self._reference, index=0)
        else:
            return self._actor

    @property
    def critic(self) -> CriticModel:  # noqa: D102
        if self._share_actor_critic:
            enable_lora(self._reference)
            return select_lora(self._reference, index=1)
        elif self._share_critic_reward:
            enable_lora(self._reward)
            self._reward.disable_bias_gain()
            return self._reward
        else:
            return self._critic

    @property
    def reference(self) -> ActorModel:  # noqa: D102
        if self._share_actor_critic:
            disable_lora(self._reference)

        return self._reference

    @property
    def reward(self) -> RewardModel:  # noqa: D102
        if self._share_critic_reward:
            disable_lora(self._reward)
            self._reward.enable_bias_gain()
        return self._reward
