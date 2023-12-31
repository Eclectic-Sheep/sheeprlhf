"""Adapted from
https://github.com/Eclectic-Sheep/sheeprl/blob/4d6e28812de97d54e64ca57e44fb6cb3d6b6c137/sheeprl/utils/registry.py
"""  # noqa: D415, D205
from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from sheeprlhf.structure import _STRUCTURED_CONFIG_GROUPS

task_registry: Dict[str, List[Dict[str, Any]]] = {}


def _register_task(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a task in the task registry."""
    # lookup containing module
    if fn.__module__ == "__main__":
        return fn
    entrypoint = fn.__name__
    module_split = fn.__module__.split(".")
    task_name = module_split[-1]
    task_type = module_split[-2]
    module = ".".join(module_split[:-1])
    registered_tasks = task_registry.get(module, None)
    new_registry = {"name": task_name, "entrypoint": entrypoint, "type": task_type}
    if registered_tasks is None:
        task_registry[module] = [new_registry]
    else:
        task_registry[module].append(new_registry)

    # add the decorated function to __all__ in algorithm
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(entrypoint)
    else:
        mod.__all__ = [entrypoint]
    return fn


def register_task():
    """Task registery decorator."""

    def inner_decorator(fn):
        return _register_task(fn)

    return inner_decorator


def auto_register_structure(cs: ConfigStore, module_name: str, node_name: str):
    """Automatically register all dataclasses in a module.

    Args:
        cs : The main ConfigStore instance.
        module_name: The name of the module to register.
        node_name: The name of the node (group) to register.
    """
    module = importlib.import_module(module_name)
    dataclasses = [obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and is_dataclass(obj)]
    for dataclass in dataclasses:
        name_available = (
            hasattr(dataclass, "config_name")
            and isinstance(dataclass.config_name, str)
            and dataclass.config_name != MISSING
        )
        if name_available:
            cs.store(group=node_name, name=dataclass.config_name, node=dataclass)


def register_structured_configs():
    """Registers all structured configs in sheeprlhf.structure."""
    cs = ConfigStore.instance()
    for key, val in _STRUCTURED_CONFIG_GROUPS.items():
        auto_register_structure(cs, f"sheeprlhf.structure.{key}", val)
