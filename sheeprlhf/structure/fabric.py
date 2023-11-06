from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FabricConfig:
    _target_: str = "lightning.fabric.Fabric"
    config_name: str = "cpu"
    devices: int = 1
    num_nodes: int = 1
    strategy: Any = "auto"
    accelerator: Any = "cpu"
    precision: Any = "32-true"
    callbacks: Optional[Any] = None


@dataclass
class AutoCudaConfig(FabricConfig):
    config_name: str = "auto_cuda"
    strategy: str = "auto"
    devices: int = 1
    num_nodes: int = 1
    accelerator: str = "cuda"
