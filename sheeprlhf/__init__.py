from sheeprlhf.task.train import dpo as dpo
from sheeprlhf.task.train import ppo as ppo
from sheeprlhf.task.train import rm as rm
from sheeprlhf.task.train import sft as sft

__version__ = "0.1.0"


def get_version():
    """Returns the version of the package."""
    return __version__
