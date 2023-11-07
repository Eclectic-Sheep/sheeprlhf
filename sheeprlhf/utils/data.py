import os
import shutil
from dataclasses import asdict
from typing import Any, Dict

import lightning
from omegaconf import OmegaConf
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer

from sheeprlhf.data import DataProcessor
from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.utils.hydra import instantiate_from_config


def prepare_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    """Creates tokenizer from Huggingface transformers library."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    special_tokens = tokenizer.special_tokens_map
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "bos_token" not in special_tokens:
        # we don't resize the tokenizer here because we want to keep the original vocab size
        # However, we need something to represent the start of the text
        # we use <|startoftext|> from gptj
        # or if they are the same, we use another word to represent the start of the text
        # this is useful for gpt2 models where bos_token and eos_token are the same
        tokenizer.bos_token = "<|startoftext|>"
    return tokenizer


def prepare_generation_config(
    tokenizer: PreTrainedTokenizer, model_cfg: ModelConfig, gen_cfg: GenConfig, fabric: lightning.Fabric
) -> Dict[str, Any]:
    """Creates generation config for Hugginface models.

    In this function, we try to solve token problems for different models.
    """
    gen_cfg_dict = asdict(gen_cfg)
    try:
        generation_config = GenerationConfig.from_pretrained(model_cfg.repo_name, **gen_cfg_dict)
    except EnvironmentError:
        # If the model does not have `generation_config.json` file, we create from scratch
        fabric.print("`generation_config.json` not found, creating `GenerationConfig` from scratch")
        generation_config = GenerationConfig(**gen_cfg_dict)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.bos_token_id = tokenizer.bos_token_id
    return generation_config


def validate_dataset(fabric: lightning.Fabric, data_cfg: DataConfig) -> DataProcessor:
    """Dataset validator.

    Validates the dataset for checking if it is required to re-create
    all preprocessing steps using tokenizers.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    data_processor: DataProcessor = instantiate_from_config(data_cfg)
    full_path = data_processor.full_path
    create_dataset: bool = True
    if os.path.isdir(full_path):
        config_path = full_path / "config.yaml"
        if not config_path.exists():
            fabric.print(f"Config file not found at {config_path} for the given dataset {data_cfg.config_name}")
            fabric.print("Dataset will be recreated and previous files will be deleted.")
        else:
            open_config = OmegaConf.load(config_path)
            loaded_dataset_cfg = DataConfig(**open_config)
            current_tokenizer = prepare_tokenizer(data_cfg.tokenizer_name)
            loaded_tokenizer = prepare_tokenizer(loaded_dataset_cfg.tokenizer_name)

            if type(current_tokenizer) != type(loaded_tokenizer):
                fabric.print("Tokenizer type changed.")
                fabric.print(f"Was {type(loaded_tokenizer)} now {type(current_tokenizer)}")
                fabric.print("New dataset will be recreated and previous files will be deleted.")
                create_dataset = True
            elif data_cfg != loaded_dataset_cfg:
                diffs = {}
                for k, v in asdict(data_cfg).items():
                    if v != getattr(loaded_dataset_cfg, k):
                        diffs[k] = (v, getattr(loaded_dataset_cfg, k))
                fabric.print("Dataset config changed.")

                fabric.print("\n".join([f"{k} was {v[0]} now {v[1]}" for k, v in diffs.items()]))
                fabric.print("New dataset will be recreated and previous files will be deleted.")
                create_dataset = True
            else:
                fabric.print("Dataset already exists. Skipping dataset creation.")
                create_dataset = False
        if create_dataset:
            shutil.rmtree(full_path)
    # This disables FastTokenizer's parallelism for multiprocessing with dataloaders
    # TODO: check if can be avoided
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    data_processor.tokenizer = prepare_tokenizer(data_cfg.tokenizer_name)
    if create_dataset and fabric.is_global_zero:
        fabric.print(f"Creating new dataset in {full_path}")
        data_processor.process()
        OmegaConf.save(data_cfg, full_path / "config.yaml")
    fabric.barrier()

    return data_processor
