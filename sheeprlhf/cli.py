"""Adapted from
https://github.com/Eclectic-Sheep/sheeprl/blob/4d6e28812de97d54e64ca57e44fb6cb3d6b6c137/sheeprl/cli.py.
"""  # noqa: D205
import importlib
import sys
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from dotenv import load_dotenv
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf

from sheeprlhf.structure.task import TASK_TYPE
from sheeprlhf.utils.hydra import instantiate_from_config
from sheeprlhf.utils.logger import print_config
from sheeprlhf.utils.registry import register_structured_configs, task_registry
from sheeprlhf.utils.structure import dotdict


def validate_args(args: Tuple[str, ...]):
    """Check if the given arguments are valid.

    They should be in the form of: `python -m sheeprlhf <run_type> <other-configs>`.
    """
    possible_args = list(TASK_TYPE.__dict__.values())
    if len(args) < 2:
        raise Exception(f"Please specify a run type. Possible run arguments: {possible_args}")

    run_type = args[1]
    if run_type not in possible_args:
        raise Exception(f"Invalid run type ({run_type}). Possible run arguments: {possible_args}")
    stripped_args = args[0:1] + args[2:]
    sys.argv = stripped_args
    return run_type


def validate_task(cfg: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    """Validate the task name and entrypoint.

    It check if the given task name is registered in the task registry.
    """
    module: Optional[str] = None
    entrypoint: Optional[str] = None
    task_name: Optional[str] = cfg.task.config_name
    for _module, _tasks in task_registry.items():
        for _task in _tasks:
            if task_name == _task["name"]:
                module = _module
                entrypoint = _task["entrypoint"]
                break
    if module is None:
        raise RuntimeError(f"Given the task named '{task_name}', no module has been found to be imported.")
    if entrypoint is None:
        raise RuntimeError(
            f"Given the module and task named '{module}' and '{task_name}' respectively, "
            "no entrypoint has been found to be imported."
        )
    return task_name, entrypoint, module


def execute(task_name: str, entrypoint: str, module: str, cfg: Dict[str, Any]):
    """Execute the given task."""
    task = importlib.import_module(f"{module}.{task_name}")
    command = task.__dict__[entrypoint]
    strategy = cfg.fabric.pop("strategy", "auto")
    fabric: Fabric = instantiate_from_config(cfg.fabric, strategy=strategy, _convert_="all")
    fabric.launch(command, cfg)


def run():
    """Run everything with hydra."""
    torch.set_float32_matmul_precision("high")
    task_type = validate_args(sys.argv)
    register_structured_configs()
    load_dotenv()

    @hydra.main(version_base="1.3", config_path="config", config_name=task_type)
    def _run(cfg: DictConfig):
        """SheepRLHF zero-code command line utility."""
        task_name, entrypoint, module = validate_task(cfg)
        print_config(cfg)
        cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        execute(task_name=task_name, entrypoint=entrypoint, module=module, cfg=cfg)

    _run()
