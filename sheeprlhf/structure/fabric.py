from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FabricConfig:
    """The default configuration for the Fabric Launcher."""

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
    """The default configuration for the Fabric Launcher with CUDA.

    It uses single GPU by default and all othersettings are set to auto.
    """

    config_name: str = "auto_cuda"
    strategy: str = "auto"
    devices: int = 1
    num_nodes: int = 1
    accelerator: str = "cuda"
