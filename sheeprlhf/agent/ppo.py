from typing import Optional

import torch

from sheeprlhf.model.actor import ActorModel
from sheeprlhf.model.critic import CriticModel
from sheeprlhf.model.reward import RewardModel
from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.structure.task import PPOConfig
from sheeprlhf.utils.helper import trainable_parameter_summary
from sheeprlhf.utils.lora import add_lora, add_multiple_lora, disable_lora, enable_lora, select_lora
from sheeprlhf.utils.model import get_model_checkpoint


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
    _init_critic_with_reward: bool

    def __init__(self, model_cfg: ModelConfig, task_cfg: PPOConfig) -> None:
        self.model_cfg = model_cfg
        self._init_critic_with_reward = task_cfg.init_critic_with_reward

        self._sft_model_cfg, self._sft_checkpoint_path = get_model_checkpoint(
            task_cfg.sft_experiment_dir, task_cfg.sft_model_name
        )
        sft_model_name = self._sft_model_cfg.repo_name

        self._rm_model_cfg, self._rm_checkpoint_path = get_model_checkpoint(
            task_cfg.rm_experiment_dir, task_cfg.sft_model_name
        )
        rm_model_name = self._rm_model_cfg.repo_name

        self._reference = ActorModel(model_cfg=self._sft_model_cfg)
        self._reward = RewardModel(model_cfg=self._rm_model_cfg)

        self._same_actor_critic = sft_model_name == rm_model_name
        self._finetune_mode = model_cfg.finetune_mode
        self._lora_enabled = self._finetune_mode == FINETUNE_MODE.LORA
        if not self._init_critic_with_reward:
            if not (self._lora_enabled and self._same_actor_critic):
                # Actor and critic cannot be shared, we fallback to the default behavior
                self._actor = ActorModel(model_cfg=self._sft_model_cfg)
                self._critic = CriticModel(model_cfg=self._sft_model_cfg)
            else:
                self._share_actor_critic = True

        else:
            if not self._lora_enabled:
                self._actor = ActorModel(model_cfg=self._sft_model_cfg)
                self._critic = CriticModel(model_cfg=self._rm_model_cfg)
            else:
                self._share_critic_reward = True

    def load_checkpoint(self, device: torch.device) -> None:
        """Load checkpoints for Actor, Critic and Reward models."""
        self._reference.load_checkpoint(
            path=self._sft_checkpoint_path, device=device, model_cfg=self._sft_model_cfg, freeze=True
        )
        self._reward.load_checkpoint(
            path=self._rm_checkpoint_path, device=device, model_cfg=self._rm_model_cfg, freeze=True
        )
        if not self._init_critic_with_reward:
            if not (self._lora_enabled and self._same_actor_critic):
                # Actor and critic cannot be shared, we fallback to the default behavior
                self._actor.load_checkpoint(
                    path=self._sft_checkpoint_path, device=device, model_cfg=self._sft_model_cfg, freeze=True
                )
                self._critic.load_checkpoint(
                    path=self._sft_checkpoint_path, device=device, model_cfg=self._sft_model_cfg, freeze=True
                )
        else:
            if not self._lora_enabled:
                self._critic.load_checkpoint(
                    path=self._rm_checkpoint_path, device=device, model_cfg=self._rm_model_cfg, freeze=True
                )
                self._actor.load_checkpoint(
                    path=self._sft_checkpoint_path, device=device, model_cfg=self._sft_model_cfg, freeze=True
                )

    def setup_finetuning(self, model_cfg: Optional[ModelConfig] = None) -> None:
        """Setup finetuning for Actor, Critic and Reward models."""
        if model_cfg is None:
            model_cfg = self.model_cfg
        lora_cfg = self.model_cfg.lora_cfg
        if not self._init_critic_with_reward:
            if self._lora_enabled and self._same_actor_critic:
                # here we can share reference model between Actor and Critic
                add_multiple_lora(self._reference, lora_cfg=lora_cfg, num=2)
            else:
                # Actor and critic cannot be shared, we fallback to the default behavior
                self._actor.setup_finetuning(model_cfg=model_cfg)
                self._critic.setup_finetuning(model_cfg=model_cfg)
        else:
            if self._lora_enabled:
                add_lora(self._reward, lora_cfg=lora_cfg)
                add_lora(self._reference, lora_cfg=lora_cfg)
            else:
                self._critic.setup_finetuning(model_cfg=model_cfg)
                self._actor.setup_finetuning(model_cfg=model_cfg)
        trainable_parameter_summary(self.actor, show_names=False, tag="Actor")
        trainable_parameter_summary(self.critic, show_names=False, tag="Critic")

    @property
    def share_actor_critic(self) -> bool:
        """Whether Actor and Critic models are shared."""
        return self._share_actor_critic

    @property
    def share_critic_reward(self) -> bool:
        """Whether Critic and Reward models are shared."""
        return self._share_critic_reward

    @property
    def lora_enabled(self) -> bool:
        """Whether LoRA is enabled."""
        return self._lora_enabled

    @property
    def actor(self) -> ActorModel:  # noqa: D102
        if self._share_actor_critic:
            enable_lora(self._reference)
            return select_lora(self._reference, index=0)
        elif self._lora_enabled and self._init_critic_with_reward:
            enable_lora(self._reference)
            return self._reference
        else:
            return self._actor

    @actor.setter
    def actor(self, actor: ActorModel) -> None:
        if self._lora_enabled and (self._share_actor_critic or self._init_critic_with_reward):
            self._reference = actor
        else:
            self._actor = actor

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

    @critic.setter
    def critic(self, critic: CriticModel) -> None:
        if self._share_actor_critic:
            self._reference = critic
        elif self._share_critic_reward:
            self._reward = critic
        else:
            self._critic = critic

    @property
    def reference(self) -> ActorModel:  # noqa: D102
        if self._share_actor_critic and self._lora_enabled:
            disable_lora(self._reference)

        return self._reference

    @reference.setter
    def reference(self, reference: ActorModel) -> None:
        self._reference = reference

    @property
    def reward(self) -> RewardModel:  # noqa: D102
        if self._share_critic_reward:
            disable_lora(self._reward)
            self._reward.enable_bias_gain()
        return self._reward

    @reward.setter
    def reward(self, reward: RewardModel) -> None:
        self._reward = reward
