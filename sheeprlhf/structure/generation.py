from dataclasses import dataclass


@dataclass
class GenConfig:
    """The default configuration for the generator."""

    # We cannot call this GenerationConfig because it will
    # conflict with transformers.GenerationConfig
    config_name: str = "default"
    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 1.0
    num_return_sequences: int = 1
