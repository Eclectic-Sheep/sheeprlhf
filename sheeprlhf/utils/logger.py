from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import lightning
import rich.syntax
import rich.tree
import torch
from lightning import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("task", "data", "model", "fabric"),
    resolve: bool = True,
    cfg_save_path: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config: Configuration composed by Hydra.
        fields: Determines which main fields from config will
            be printed and in what order.
        resolve: Whether to resolve reference fields of DictConfig.
        cfg_save_path: Path to save the config tree.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if cfg_save_path is not None:
        with open(os.path.join(os.getcwd(), "config_tree.txt"), "w") as fp:
            rich.print(tree, file=fp)


@rank_zero_only
def log_text(fabric: lightning.Fabric, text: str, name: str, step: int):
    """Wrapper function to log text to tensorboard."""
    if fabric.logger is not None:
        if isinstance(fabric.logger, lightning.fabric.loggers.tensorboard.TensorBoardLogger):
            fabric.logger.experiment.add_text(name, text, step)
        else:
            warnings.warn(f"Logging text is not supported for {type(fabric.logger)}", stacklevel=2)


def trainable_parameter_summary(
    model: torch.nn.Module, show_names: bool = False, fabric: Optional[lightning.Fabric] = None
):
    """Prints a summary of the trainable parameters of a model."""
    print_fn = fabric.print if fabric is not None else print
    trainable = {"int8": 0, "bf16": 0, "fp16": 0, "fp32": 0, "other": 0}
    non_trainable = {"int8": 0, "bf16": 0, "fp16": 0, "fp32": 0, "other": 0}
    param_count = {"trainable": trainable, "non_trainable": non_trainable}
    trainable_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            dict_name = "trainable"
            trainable_param_names.append(name)
        else:
            dict_name = "non_trainable"
        num_params = param.numel()
        if param.dtype == torch.int8:
            param_count[dict_name]["int8"] += num_params
        elif param.dtype == torch.bfloat16:
            param_count[dict_name]["bf16"] += num_params
        elif param.dtype == torch.float16:
            param_count[dict_name]["fp16"] += num_params
        elif param.dtype == torch.float32:
            param_count[dict_name]["fp32"] += num_params
        else:
            param_count[dict_name]["other"] += num_params

    if show_names:
        print_fn("Trainable parameter names:")
        print_fn(trainable_param_names)
    print_fn("Parameter Statistics:")
    print_fn(f"Trainable {trainable}")
    print_fn(f"Non-Trainable {non_trainable}")
    total_params = sum([sum(v.values()) for v in param_count.values()])
    total_trainable_params = sum([v for _, v in param_count["trainable"].items()])
    print_fn(
        f"Total: {total_params}, "
        f"Trainable: {total_trainable_params}, "
        f"Percentage: {total_trainable_params/total_params:.2%}"
    )


def create_tensorboard_logger(
    fabric: Fabric, cfg: Dict[str, Any], override_log_level: bool = False
) -> Tuple[Optional[TensorBoardLogger]]:
    """Creates tensorboard logger.

    Set logger only on rank-0 but share the logger directory: since
    we don't know. what is happening during the `fabric.save()` method,
    at least we assure that all ranks save under the same named folder.
    As a plus, rank-0 sets the time uniquely for everyone.
    """
    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    logger = None
    if fabric.is_global_zero:
        root_dir = os.path.join("logs", "runs", cfg.root_dir)
        if override_log_level or cfg.metric.log_level > 0:
            logger = TensorBoardLogger(root_dir=root_dir, name=cfg.run_name)
    return logger


def get_log_dir(fabric: Fabric, root_dir: str, run_name: str, share: bool = True) -> str:
    """Return and, if necessary, create the log directory.

    If there are more than one processes, the rank-0 process shares
    the directory to the others
    (if the `share` parameter is set to `True`).

    Args:
        fabric: the fabric instance.
        root_dir: the root directory of the experiment.
        run_name: the name of the experiment.
        share: whether or not to share the `log_dir` among processes.

    Returns:
        The log directory of the experiment.
    """
    world_collective = TorchCollective()
    if fabric.world_size > 1 and share:
        world_collective.setup()
        world_collective.create_group()
    if fabric.is_global_zero:
        # If the logger was instantiated, then take the log_dir from it
        if len(fabric.loggers) > 0:
            log_dir = fabric.logger.log_dir
        else:
            # Otherwise the rank-zero process creates the log_dir
            save_dir = os.path.join("logs", "runs", root_dir, run_name)
            fs = get_filesystem(root_dir)
            try:
                listdir_info = fs.listdir(save_dir)
                existing_versions = []
                for listing in listdir_info:
                    d = listing["name"]
                    bn = os.path.basename(d)
                    if _is_dir(fs, d) and bn.startswith("version_"):
                        dir_ver = bn.split("_")[1].replace("/", "")
                        existing_versions.append(int(dir_ver))
                version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
                log_dir = os.path.join(save_dir, f"version_{version}")
            except OSError:
                warnings.warn("Missing logger folder: %s", save_dir, stacklevel=2)
                log_dir = os.path.join(save_dir, f"version_{0}")

            os.makedirs(log_dir, exist_ok=True)
        if fabric.world_size > 1 and share:
            world_collective.broadcast_object_list([log_dir], src=0)
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
    return log_dir


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """Wrapper around `print` that only prints from rank 0."""
    print(*args, **kwargs)
    if kwargs.get("file") is None:
        with open("rank_zero_log.txt", "a") as f:
            print(*args, **kwargs, file=f)
