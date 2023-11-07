import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import II, MISSING


@dataclass
class DataConfig:
    """The main class for processing data for the RLHF algorithm.

    Args:
        config_name: The name of the data configuration.
        dataset_name: The name of the dataset to load.
        root_dir: The directory where the processed data will be saved.
        tokenizer_name: The name of the tokenizer to use.
        max_length: The maximum length of the input tokens. Defaults to 512.
        max_prompt_length: The maximum length of the prompt tokens. Defaults to 512.
        num_samples: The number of samples to use. Defaults to None.
        ignore_index: The index to use for ignored tokens. Defaults to -1.
        remove_same_responses: Whether to remove samples with the same response. Defaults to True.
        remove_same_inputs: Whether to remove samples with the same input. Defaults to True.
        minimum_response_length: The minimum length of the response tokens. Defaults to 2.
        save_skipped_examples: Whether to save skipped examples. Defaults to False.
        validation_split: The validation split. Defaults to 0.1.
        reward_model_split: The reward model split. Defaults to 0.5.
        shuffle: Whether to shuffle the dataset. Defaults to True.
        seed: The random seed. Defaults to 42.
        split_names: The names of the splits. Defaults to ("train", "val", "test").
    """

    _target_: str = "sheeprlhf.data.DataProcessor"
    config_name: str = MISSING
    dataset_name: str = MISSING
    root_dir: str = Path("./rlhf_data")
    tokenizer_name: str = II("model.repo_name")
    max_length: int = 256
    max_prompt_length: int = 128
    num_samples: Optional[int] = None
    ignore_index: int = -1
    remove_same_responses: bool = True
    remove_same_inputs: bool = True
    minimum_response_length: int = 5
    save_skipped_examples: bool = False
    shuffle: bool = True
    seed: int = II("seed")
    validation_split: float = 0.1
    reward_model_split: float = 0.5
    split_names: Tuple[str] = ("train", "test")
    debug: bool = II("debug")


@dataclass
class HelpfulHarmlessConfig(DataConfig):
    """The configuration for the HelpfulHarmless dataset available in Huggingface."""

    _target_: str = "sheeprlhf.data.HelpfulHarmlessData"
    config_name: str = "helpful_harmless"
    dataset_name: str = "Dahoas/full-hh-rlhf"


@dataclass
class SummarizationConfig(DataConfig):
    """The configuration for the OpenAI Reddit posts summarization dataset available in Huggingface."""

    _target_: str = "sheeprlhf.data.SummarizationData"
    config_name: str = "summarization"
    dataset_name: str = "CarperAI/openai_summarize_comparisons"
    max_length: int = 512
    max_prompt_length: int = 384

    def __post_init__(self) -> None:
        if self.remove_same_inputs:
            warnings.warn(
                "`remove_same_inputs` is set to True. This means only one example per duplicated "
                "input will be kept. This may result in a smaller dataset than expected "
                "because this dataset may contain many negative (rejected) examples from "
                "the same (chosen) input.",
                stacklevel=2,
            )
