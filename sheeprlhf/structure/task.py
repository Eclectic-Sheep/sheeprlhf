from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import II, MISSING


@dataclass
class TASK_TYPE:
    TRAIN: str = "train"
    EVAL: str = "eval"


# Omegaconf does not support Literal String types
class RM_LOSS_TYPE(Enum):
    AVERAGE = "average"
    LAST_TOKEN = "last_token"
    PER_SAMPLE = "per_sample"


@dataclass
class TrainTaskConfig:
    """Configuration class for RLHF algorithm.

    Attributes:
        _name_ (str): Name of the algorithm.
        epochs (int): Number of epochs for training.
        save_interval (int): Every save interval steps model will be saved.
        eval_interval (int): Every eval interval steps model will be evaluated or text will be generated.
        log_interval (int): Every log interval steps metrics will be logged to selected logger and progress
            bar will be updated.
        eval_iters (int): Number of iterations to evaluate on. If None, evaluate on full validation set.
        num_workers (int): Number of workers for dataloaders.
        mini_batch_size (int): Mini batch size for training. This is the expected batch size when there
            is no gradient accumulation applied.
        micro_batch_size (int): Micro batch size for training. If mini_batch_size // micro_batch_size == 1,
            no gradient accumulation is performed.
        gradient_clip_val (float): Gradient clipping value.
        gradient_accumulation_steps (int): Number of gradient accumulation steps. It will be calculated automatically.
        use_masked_targets (bool): Whether to use only responses for training or use both prompts and responses.
        lr_warmup_steps (int): Number of warmup steps for learning rate scheduler. Default scheduler has linear warmup.
    """

    config_name: str = MISSING
    epochs: int = 1
    save_interval: int = 1000
    eval_interval: int = 1000
    log_interval: int = 5
    eval_iters: int = 100
    num_workers: int = 4
    mini_batch_size: int = 8
    micro_batch_size: int = 8
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    use_masked_targets: bool = False
    lr_warmup_steps: int = 200


@dataclass
class SFTConfig(TrainTaskConfig):
    """Configuration class for the RLHF-SFT algorithm.

    Args:
        _name_ (str): The name of the algorithm.
        label_smoothing (float): The label smoothing value to use when training the model.
            It should be a value between 0.0 and 1.0. Default is 0.0. Label smoothing helps
            model to generalize better by preventing it from predicting with 100% confidence.
    """

    config_name: str = "sft"
    label_smoothing: float = 0.0


@dataclass
class RewardModellingConfig(TrainTaskConfig):
    """Configuration class for the RLHF-RM algorithm.

    Attributes:
        name (str): The name of the algorithm.
        loss_type (RM_LOSS_TYPE): The type of loss function to use.
        sft_experiment_dir (Optional[str]): The path to the supervised finetuning experiment directory
            to load the model from. It will be used to initialize both the actor model and the reference.
    """

    config_name: str = "rm"
    loss_type: RM_LOSS_TYPE = RM_LOSS_TYPE.PER_SAMPLE
    sft_experiment_dir: Optional[str] = II("sft_experiment_dir")

    def __post_init__(self):
        if isinstance(self.loss_type, str):
            self.loss_type = RM_LOSS_TYPE[self.loss_type.upper()]


@dataclass
class DPOConfig(TrainTaskConfig):
    """Configuration class for RLHF DPO algorithm.

    Attributes:
        _name_ (str): Name of the algorithm. Default is "rlhf_dpo".
        sft_experiment_dir (Optional[str]): Path to the experiment directory. Default is None.
        use_masked_targets (bool): Whether to use masked targets updating the policy. Default is True.
        reference_free (bool): Whether to use the reference model or not when computing the DPO loss.
            If True, we ignore reference model and implicitly use a reference model that assigns equal
            probability to all responses. Default is False.
        beta (float): Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
            We ignore the reference model when the beta is 0.0. Default is 0.1.
    """

    config_name: str = "dpo"
    sft_experiment_dir: Optional[str] = II("sft_experiment_dir")
    use_masked_targets: bool = True
    reference_free: bool = False
    beta: float = 0.1


@dataclass
class PPOConfig(TrainTaskConfig):
    """Configuration class for PPO algorithm.

    Attributes:
        _name_ (str): Name of the algorithm. Default is "rlhf_ppo".
        rollout_size (int): Rollout size for PPO. For every training iteration this number of samples will
            be sampled from dataset and each will be used for generating response.
        rollout_mini_batch_size (int): Rollout mini batch size for PPO. This number is useful when the
            GPU memory is not sufficient for running all generation code with single batch.
        ppo_epochs (int): Number of ppo epochs to training. `ppo_step` function will be called `ppo_epochs` times
        normalize_rewards (bool): Whether to whiten rewards
        normalize_advantages (bool): Whether to whiten advantages
        adaptive_kl_coeff (bool): Whether to use adaptively changing KL divergence coefficient
        clip_rewards (bool): Whether to clip rewards
        reward_clip_value (float): Reward clipping value
        init_kl_coeff (float): KL divergence coefficient for comparing actor model with reference model.
            Higher value means more trust to reference model.
        target_kl_coeff (float): Target KL divergence coefficient
        clip_coeff (float): Clip coefficient for PPO loss
        vf_coeff (float): Value loss coefficient for PPO loss
        gae_gamma (float): Discount factor for GAE(Generalized Advantage Estimation)
        gae_lambd (float): Lambda for GAE(Generalized Advantage Estimation)
        sft_experiment_dir (str): Path to supervised finetuning experiment directory. Latest checkpoint will be loaded.
        rm_experiment_dir (str): Path to reward modelling experiment directory. Latest checkpoint will be loaded.
        actor_learning_rate (float): Learning rate for actor optimizer
        critic_learning_rate (float): Learning rate for critic optimizer
        init_critic_with_rm (bool): Whether to initialize critic with reward model checkpoint or not.
    """

    config_name: str = "ppo"
    rollout_size: int = 128
    rollout_mini_batch_size: int = 32
    ppo_epochs: int = 1
    normalize_rewards: bool = True
    normalize_advantages: bool = True
    adaptive_kl_coeff: bool = False
    clip_rewards: bool = True
    reward_clip_value: float = 5.0
    init_kl_coeff: float = 0.1
    target_kl_coeff: float = 0.1
    clip_coeff: float = 0.2
    vf_coeff: float = 0.1
    gae_gamma: float = 1.0
    gae_lambd: float = 0.95
    sft_experiment_dir: str = II("sft_experiment_dir")
    rm_experiment_dir: str = II("rm_experiment_dir")
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 1e-6
    init_critic_with_rm: bool = True
