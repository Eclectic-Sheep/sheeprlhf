import platform

from lightning_utilities.core.imports import RequirementCache

_IS_EVALUATE_AVAILABLE = RequirementCache("evaluate")
_IS_GRADIO_AVAILABLE = RequirementCache("gradio")
_IS_LIT_GPT_AVAILABLE = RequirementCache("lit-gpt")
_IS_WINDOWS = platform.system() == "Windows"
_IS_MACOS = platform.system() == "Darwin"
