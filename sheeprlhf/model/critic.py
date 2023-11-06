import os

import lightning
import torch
from lightning_utilities.core.rank_zero import rank_zero_only

from sheeprlhf.model.finetune import FinetuneModel
from sheeprlhf.structure.model import FINETUNE_MODE, ModelConfig
from sheeprlhf.utils.lora import add_lora, get_lora_state_dict, merge_lora
from sheeprlhf.utils.model import load_hf_transformer


class CriticModel(FinetuneModel):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        model = load_hf_transformer(model_cfg)
        transformer_config = model.base_model.config
        if model_cfg.embedding_dim_name is None:
            if hasattr(model, "get_input_embeddings"):
                embedding_dim = model.get_input_embeddings().weight.shape[-1]
            else:
                raise ValueError("embedding_dim_name is None and model does not have `get_input_embeddings` method")
        else:
            embedding_dim = getattr(transformer_config, model_cfg.embedding_dim_name, None)
            if embedding_dim is None:
                raise ValueError(
                    f"`embedding_dim_name={model_cfg.embedding_dim_name}` not found in "
                    "`transformer_config` from hugginface library"
                )
        if model_cfg.transformer_name is None:
            # If any transformer name is not provided, we search for common attribute names usually
            # avaliable inside huggingface library or already tested model's attribute names.
            model_type_str = str(type(model))
            if hasattr(model, "transformer"):
                self.transformer = model.transformer
            elif hasattr(model, "model"):
                self.transformer = model.model
            elif hasattr(model, "layers") and "MixFormerSequentialForCausalLM" in model_type_str:
                # it is the phi model we remove the last layer
                model.layers = model.layers[:-1]
                self.transformer = model
            else:
                raise ValueError(
                    f"{model} Could not find transformer, searched for 'transformer' and 'model' attributes, "
                    "if your model has a different attribute name, please specify it in `transformer_name`"
                )
        else:
            self.transformer = getattr(model, model_cfg.transformer_name)

        self.head = torch.nn.Linear(embedding_dim, 1, bias=False)
        self.head.apply(self.init_normal)

    def setup_finetuning(self, fabric: lightning.Fabric):
        super().setup_finetuning(fabric)
        if self.model_cfg.finetune_mode == FINETUNE_MODE.LORA:
            for param in self.head.parameters():
                param.requires_grad = True

    def init_normal(self, module):
        if type(module) == torch.nn.Linear:
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, **kwargs):
        if self.training and not self.model_cfg.use_attention_mask:
            kwargs.pop("attention_mask")
        out = self.transformer(**kwargs)[0]
        value = self.head(out)
        return value.squeeze(-1)

    def get_head_state_dict(self):
        sd = self.state_dict()
        sd = {k: v for k, v in sd.items() if "head" in k}
        return sd

    @classmethod
    def load_checkpoint(
        self,
        path: str,
        fabric: lightning.Fabric,
        model_cfg: ModelConfig,
        freeze: bool = False,
    ):
        sd = torch.load(path, map_location=fabric.device)
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("model.model", "transformer")
            new_k = new_k.replace("model.", "")
            new_sd[new_k] = v
        if model_cfg.finetune_mode == FINETUNE_MODE.LORA:
            add_lora(self.model, lora_cfg=model_cfg.lora_cfg)
            self.model.load_state_dict(new_sd, strict=False)
            merge_lora(self.model)
        else:
            self.model.load_state_dict(new_sd)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @rank_zero_only
    def save_checkpoint(self, fabric: lightning.Fabric, experiment_dir: str, model_cfg: ModelConfig, step):
        output_file = os.path.join(experiment_dir, "model", f"checkpoint-{step}.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if model_cfg.finetune_mode == FINETUNE_MODE.LORA:
            sd = get_lora_state_dict(self)
            head_sd = self.get_head_state_dict()
            sd.update(head_sd)
        else:
            sd = self.state_dict()
        fabric.save(output_file, sd)


class RewardModel(CriticModel):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__(model_cfg)
        self.gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self._disable_bias_gain = False

    def disable_bias_gain(self):
        self._disable_bias_gain = True

    def enable_bias_gain(self):
        self._disable_bias_gain = False

    def forward(self, **kwargs):
        value_out = super().forward(**kwargs)
        if self._disable_bias_gain:
            return value_out
        return value_out * self.gain + self.bias

    def get_head_state_dict(self):
        head_state_dict = super().get_head_state_dict()
        if not self._disable_bias_gain:
            head_state_dict.update({"gain": self.gain, "bias": self.bias})
        return head_state_dict
