from typing import Dict

import torch

from sheeprlhf.model.critic import CriticModel
from sheeprlhf.structure.model import ModelConfig


class RewardModel(CriticModel):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__(model_cfg=model_cfg)
        self.gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self._disable_bias_gain = False

    def disable_bias_gain(self):
        self._disable_bias_gain = True

    def enable_bias_gain(self):
        self._disable_bias_gain = False

    def forward(self, **kwargs) -> torch.Tensor:
        value_out = super().forward(**kwargs)
        if self._disable_bias_gain:
            return value_out
        return value_out * self.gain + self.bias

    def get_head_state_dict(self) -> Dict[str, torch.Tensor]:
        head_state_dict = super().get_head_state_dict()
        if not self._disable_bias_gain:
            head_state_dict.update({"gain": self.gain, "bias": self.bias})
        return head_state_dict
