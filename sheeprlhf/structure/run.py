from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING, SI

from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.fabric import FabricConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.structure.optim import OptimizerConfig
from sheeprlhf.structure.task import EvalTaskConfig, TrainTaskConfig

train_defaults = [
    {"task": "???"},
    {"model": "???"},
    {"data": "???"},
    {"fabric": "cpu"},
    {"optim": "adamw"},
    {"experiment": None},
]
eval_defaults = [
    {"task": "???"},
    {"fabric": "cpu"},
]


@dataclass
class TrainRunConfig:  # noqa: D101
    config_name: str = "base_train"
    defaults: List[Any] = field(default_factory=lambda: train_defaults)

    task: TrainTaskConfig = MISSING
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    fabric: FabricConfig = MISSING
    optim: OptimizerConfig = MISSING

    # Default configs
    generation: GenConfig = GenConfig()
    experiment: Any = None
    dry_run: bool = False
    seed: int = 42
    torch_deterministic: bool = False
    run_name: str = SI("seed_${seed}")
    root_dir: str = SI("${data.config_name}/${model.config_name}/${task.config_name}/${now:%Y-%m-%d_%H-%M-%S}")


@dataclass
class EvalRunConfig:  # noqa: D101
    config_name: str = "base_eval"
    defaults: List[Any] = field(default_factory=lambda: eval_defaults)
    # Mandatory configs
    task: EvalTaskConfig = MISSING
    fabric: FabricConfig = MISSING

    # Default configs
    generation: GenConfig = GenConfig()
    dry_run: bool = False
    seed: int = 42
    experiment: Any = None
