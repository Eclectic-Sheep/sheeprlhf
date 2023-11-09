from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, SI

from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.fabric import AutoCudaConfig, FabricConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.structure.optim import AdamWConfig, OptimizerConfig
from sheeprlhf.structure.task import TrainTaskConfig


@dataclass
class TrainRunConfig:  # noqa: D101
    config_name: str = "base_train"
    # Mandatory configs
    task: TrainTaskConfig = MISSING
    model: ModelConfig = MISSING
    data: DataConfig = MISSING

    # Default configs
    fabric: FabricConfig = AutoCudaConfig()
    optim: OptimizerConfig = AdamWConfig()
    generation: GenConfig = GenConfig()
    experiment: Any = None
    dry_run: bool = False
    seed: int = 42
    torch_deterministic: bool = False
    run_name: str = SI("seed_${seed}")
    root_dir: str = SI("${data.config_name}/${model.config_name}/${task.config_name}/${now:%Y-%m-%d_%H-%M-%S}")
