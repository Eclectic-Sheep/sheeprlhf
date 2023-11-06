import os
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel

from sheeprlhf.structure.model import HuggingFaceConfig, ModelConfig


def load_hf_transformer(model_cfg: ModelConfig) -> PreTrainedModel:
    """Load a HuggingFace transformer model.
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


def get_last_checkpoint_path(experiment_dir: str):
    model_dir = os.path.join(experiment_dir, "model")
    checkpoints = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[-2].split("-")[-1]))
    return checkpoints[-1]


def prepare_optimizer_parameters(model: torch.nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Taken from  https://github.com/karpathy/nanoGPT"""
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


def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().cpu().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
