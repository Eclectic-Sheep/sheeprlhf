import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel

from sheeprlhf.structure.model import HuggingFaceConfig, ModelConfig


def load_hf_transformer(model_cfg: ModelConfig) -> PreTrainedModel:
    """Load a HuggingFace transformer model.

    The function can also freeze the model at the end of function.

    Args:
        model_cfg: ModelConfig object.

    Returns:
        A HuggingFace transformer model.
    """
    model_cls = AutoModel if not model_cfg.casual else AutoModelForCausalLM
    hf_cfg: HuggingFaceConfig = model_cfg.library_cfg
    auto_config = AutoConfig.from_pretrained(model_cfg.repo_name, trust_remote_code=hf_cfg.trust_remote_code)
    if hasattr(auto_config, "dropout"):
        auto_config.dropout = 0.0 if model_cfg.disable_dropout else auto_config.dropout
    auto_config.use_cache = hf_cfg.use_cache
    auto_config.torch_dtype = torch.get_default_dtype()
    model = model_cls.from_pretrained(
        model_cfg.repo_name,
        trust_remote_code=hf_cfg.trust_remote_code,
        load_in_8bit=hf_cfg.load_in_8bit,
        low_cpu_mem_usage=hf_cfg.low_cpu_mem_usage,
        config=auto_config,
    )
    if model_cfg.freeze_transformer:
        for param in model.parameters():
            param.requires_grad = False
    return model


def get_model_checkpoint(experiment_dir: str, model_name: Optional[str] = None) -> Tuple[ModelConfig, str]:
    """It retrives model checkpoint and related model config.

    By default it will return the last checkpoint if model_name is not provided.

    Args:
        experiment_dir: Output of the trained experiment path.
        model_name: Name of the model to retrieve the checkpoint. Example: `model-1000.pt`

    Returns:
        A tuple of ModelConfig and checkpoint path.
    """
    exp_dir = Path(experiment_dir)
    model_dir = exp_dir / "model"
    exp_cfg = OmegaConf.load(exp_dir / ".hydra/config.yaml")
    model_cfg = ModelConfig(**exp_cfg.model)
    if model_name is None:
        checkpoints = [os.path.join(str(model_dir), f) for f in os.listdir(str(model_dir)) if f.endswith(".pt")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[-2].split("-")[-1]))
        selected_checkpoint = checkpoints[-1]
    else:
        selected_checkpoint = os.path.join(model_dir, model_name)
        if not os.path.exists(selected_checkpoint):
            raise FileNotFoundError(f"Checkpoint {selected_checkpoint} does not exist.")
    return model_cfg, selected_checkpoint


def prepare_optimizer_parameters(model: torch.nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Taken from  https://github.com/karpathy/nanoGPT."""
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    return optim_groups, num_decay_params, num_nodecay_params


def compute_grad_norm(model: torch.nn.Module) -> float:  # noqa: D103
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().cpu().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
