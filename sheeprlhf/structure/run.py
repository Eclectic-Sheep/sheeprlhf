from dataclasses import dataclass

from omegaconf import MISSING, SI

from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.fabric import AutoCudaConfig, FabricConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.structure.optim import AdamWConfig, OptimizerConfig
from sheeprlhf.structure.task import TrainTaskConfig


@dataclass
class TrainRunConfig:
    config_name: str = "base_train"
    # Mandatory configs
    task: TrainTaskConfig = MISSING
    model: ModelConfig = MISSING
    data: DataConfig = MISSING

    # Default configs
    fabric: FabricConfig = AutoCudaConfig()
    optim: OptimizerConfig = AdamWConfig()
    generation: GenConfig = GenConfig()

    debug: bool = False
    seed: int = 42
    torch_deterministic: bool = False
    # TODO: change the exp name to MISSING
    exp_name: str = "test_training"
    run_name: str = SI("seed_${seed}")
    root_dir: str = SI("${task.config_name}/${data.config_name}/${now:%Y-%m-%d_%H-%M-%S}")
