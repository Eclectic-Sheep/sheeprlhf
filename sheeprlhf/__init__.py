from sheeprlhf.utils.cache import _IS_EVALUATE_AVAILABLE

if _IS_EVALUATE_AVAILABLE:
    from sheeprlhf.task.eval import perplexity as perplexity
    from sheeprlhf.task.eval import rouge as rouge
from sheeprlhf.task.train import dpo as dpo
from sheeprlhf.task.train import ppo as ppo
from sheeprlhf.task.train import rm as rm
from sheeprlhf.task.train import sft as sft

__version__ = "0.1.0"


def get_version():
    """Returns the version of the package."""
    return __version__
