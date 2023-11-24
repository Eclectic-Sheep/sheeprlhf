from typing import Any, Dict

from sheeprlhf.data.base import DataProcessor


class HelpfulHarmlessData(DataProcessor):
    """Data processor for HelpfulHarmless tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_prompt(self, sample: Dict[str, Any]) -> str:  # noqa: D102
        return sample["prompt"]

    def get_chosen(self, sample: Dict[str, Any]) -> str:  # noqa: D102
        return sample["chosen"]

    def get_rejected(self, sample: Dict[str, Any]) -> str:  # noqa: D102
        return sample["rejected"]

    def wrap_prompt(self, prompt: str) -> str:  # noqa: D102
        return "\n\nHuman: " + prompt + "\n\nAssistant: "

    def get_example_prompt(self) -> str:  # noqa: D102
        return "How does the computer work?"
