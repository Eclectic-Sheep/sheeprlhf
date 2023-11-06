from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from hydra.utils import instantiate


def instantiate_from_config(config: Any, *args, **kwargs):
    config_copy = deepcopy(config)
    if is_dataclass(config_copy):
        config_copy = asdict(config_copy)
    if isinstance(config_copy, dict) and "config_name" in config_copy:
        config_copy.pop("config_name")
    return instantiate(config_copy, *args, **kwargs)
