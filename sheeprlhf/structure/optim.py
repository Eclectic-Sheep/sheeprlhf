from dataclasses import dataclass
from typing import Tuple

from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    """A generic configuration for optimizers."""

    _target_: str = MISSING
    lr: float = 1e-5


@dataclass
class Adam(OptimizerConfig):
    """The default configuration for Adam."""

    _target_: str = "torch.optim.Adam"
    config_name: str = "adam"
    lr: float = 1e-5
    betas: Tuple[float] = (0.9, 0.999)
    eps: float = 1e-5


@dataclass
class AdamWConfig(OptimizerConfig):
    """The default configuration for AdamW."""

    _target_: str = "torch.optim.AdamW"
    config_name: str = "adamw"
    lr: float = 1e-5
    betas: Tuple[float] = (0.9, 0.999)
    eps: float = 1e-6
    weight_decay: float = 0.0


@dataclass
class SGDConfig(OptimizerConfig):
    """The default configuration for SGD."""

    _target_: str = "torch.optim.SGD"
    config_name: str = "sgd"
    lr: float = 1e-05
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
