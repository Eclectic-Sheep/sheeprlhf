import json
from pathlib import Path
from typing import Any, Dict

import evaluate as eval_lib
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprlhf.data.base import TextDataset
from sheeprlhf.data.collate import EvaluateCollate
from sheeprlhf.model.casual import CasualModel
from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.task import RougeConfig
from sheeprlhf.utils.data import prepare_generation_config, validate_dataset
from sheeprlhf.utils.model import get_model_checkpoint
from sheeprlhf.utils.registry import register_task

rouge_metric = eval_lib.load("rouge")
from transformers import AutoTokenizer, GenerationConfig


@torch.inference_mode()
def evaluate(  # noqa: D103
    model: CasualModel,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    dry_run: bool = False,
):
    generated_list = []
    target_list = []
    bar_len = len(dataloader) if not dry_run else 1
    with tqdm(dataloader, total=bar_len, desc="Evaluating") as pbar:
        for batch in pbar:
            generated_input_ids = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                generation_config=generation_config,
                use_cache=True,
            )
            response_ground_truth = []
            response_generated = []
            for i in range(len(batch["input_ids"])):
                prompt_len = batch["prompt_len"][i]
                response_ground_truth.append(batch["input_ids"][i][prompt_len:])
                response_generated.append(generated_input_ids[i][prompt_len:])

            generated_response_text = tokenizer.batch_decode(response_generated, skip_special_tokens=True)
            target_response_text = tokenizer.batch_decode(response_ground_truth, skip_special_tokens=True)

            generated_list.extend(generated_response_text)
            target_list.extend(target_response_text)
            pbar.update(1)
            if dry_run:
                break
    rouge_score = rouge_metric.compute(predictions=generated_list, references=target_list)
    return rouge_score


@register_task()
def main(fabric: Fabric, cfg: Dict[str, Any]):  # noqa: D103
    seed = cfg.seed
    dry_run = cfg.dry_run
    task_cfg = RougeConfig(**cfg.task)
    gen_cfg = GenConfig(**cfg.generation)
    data_split = task_cfg.data_split
    mini_batch_size = task_cfg.mini_batch_size
    num_workers = task_cfg.num_workers

    fabric.seed_everything(seed=seed + fabric.global_rank)

    experiment_dir = task_cfg.experiment_dir
    exp_cfg = OmegaConf.load(Path(experiment_dir) / ".hydra/config.yaml")
    data_cfg: DataConfig = DataConfig(**exp_cfg.data)

    # Setup Model
    ckpt_model_cfg, checkpoint_path = get_model_checkpoint(experiment_dir, task_cfg.model_name)
    with fabric.init_module(empty_init=ckpt_model_cfg.fabric_empty_init):
        model = CasualModel(model_cfg=ckpt_model_cfg)
        model.load_checkpoint(checkpoint_path, device=fabric.device, model_cfg=ckpt_model_cfg, freeze=True)
    model = fabric.setup_module(model)
    model.eval()

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

    # Setup Dataloaders
    collator = EvaluateCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    dataset = TextDataset(dataframe_path=dataset_path / f"finetune_{data_split}.pkl")
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=mini_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
    )
    dataloader = fabric.setup_dataloaders(dataloader)

    # Setup Generation Config
    eval_generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=ckpt_model_cfg,
        gen_cfg=gen_cfg,
        fabric=fabric,
    )

    result = evaluate(
        model=model,
        generation_config=eval_generation_config,
        dataloader=dataloader,
        tokenizer=tokenizer,
        dry_run=dry_run,
    )
    if fabric.is_global_zero:
        eval_folder = Path(experiment_dir) / "eval"
        eval_folder.mkdir(parents=True, exist_ok=True)
        eval_file = eval_folder / f"rouge_{data_split}.json"
        with eval_file.open("w") as f:
            json.dump(result, f)
    fabric.print(f"Rouge score on {data_cfg.dataset_name} {data_split} set: {result}")
