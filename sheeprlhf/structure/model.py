from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import II, MISSING


# Omegaconf does not support Literal String types
class FINETUNE_MODE(Enum):
    """Finetuning mode for models."""

    ALL = "ALL"
    LORA = "LORA"


@dataclass
class LibraryConfig:
    """A generic configuration for model loading libraries."""

    model_name: str = MISSING


@dataclass
class HuggingFaceConfig(LibraryConfig):
    """Extension configuration for Huggingface based models."""

    model_name: str = II("model.repo_name")
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    low_cpu_mem_usage: bool = False
    use_cache: bool = False


@dataclass
class LORAConfig:
    """Configurations for LORA finetuning."""

    targets: str = MISSING
    rank: int = 16
    alpha: float = 16
    dropout: float = 0.0


@dataclass
class ModelConfig:
    """A generic configuration for models."""

    config_name: str = MISSING
    repo_name: Optional[str] = None
    embedding_dim_name: Optional[str] = None
    transformer_name: Optional[str] = None
    casual: bool = True
    freeze_transformer: bool = False
    disable_dropout: bool = False
    library_cfg: HuggingFaceConfig = HuggingFaceConfig()
    finetune_mode: FINETUNE_MODE = FINETUNE_MODE.ALL
    lora_cfg: Optional[LORAConfig] = None
    use_attention_mask: bool = True
    fabric_empty_init: bool = True

    def __post_init__(self):
        if isinstance(self.finetune_mode, str):
            self.finetune_mode = FINETUNE_MODE(self.finetune_mode)


@dataclass
class OPTConfig(ModelConfig):
    """Configurations for OPT based models."""

    config_name: str = "opt"
    repo_name: str = "facebook/opt-350m"
    embedding_dim_name: Optional[str] = "word_embed_proj_dim"
    lora_cfg: Optional[LORAConfig] = LORAConfig(targets="('q_proj','v_proj')")


@dataclass
class GPT2Config(ModelConfig):
    """Configurations for GPT2 based models."""

    config_name: str = "gpt2"
    repo_name: str = "gpt2-medium"
    embedding_dim_name: Optional[str] = "n_embd"
    lora_cfg: Optional[LORAConfig] = LORAConfig(targets="('c_attn',)")


@dataclass
class PhiConfig(ModelConfig):
    """Configurations for Phi based models."""

    config_name: str = "phi"
    repo_name: str = "microsoft/phi-1_5"
    library_cfg: HuggingFaceConfig = HuggingFaceConfig(trust_remote_code=True)
    lora_cfg: Optional[LORAConfig] = LORAConfig(targets="('Wqkv','out_proj')")
    # This model cannot use attention mask during the training
    # https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_mixformer_sequential.py#L756
    use_attention_mask: bool = False


@dataclass
class FalconConfig(ModelConfig):
    """Configurations for Falcon based models."""

    config_name: str = "falcon"
    repo_name: str = "tiiuae/falcon-7b"
    library_cfg: HuggingFaceConfig = HuggingFaceConfig(trust_remote_code=True)
    lora_cfg: Optional[LORAConfig] = LORAConfig(targets="('query_key_value',)")
