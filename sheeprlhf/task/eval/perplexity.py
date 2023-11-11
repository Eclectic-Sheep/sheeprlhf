import json
from pathlib import Path
from typing import Any, Dict

import torch
from lightning import Fabric
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprlhf.data.base import TextDataset
from sheeprlhf.data.collate import EvaluateCollate
from sheeprlhf.loss.sft import finetune_loss
from sheeprlhf.model.casual import CasualModel
from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.task import PerplexityConfig
from sheeprlhf.utils.data import validate_dataset
from sheeprlhf.utils.model import get_model_checkpoint
from sheeprlhf.utils.registry import register_task


@torch.inference_mode()
def evaluate(  # noqa: D103
    model: CasualModel,
    use_masked_targets: bool,
    label_smoothing: float,
    data_cfg: DataConfig,
    dataloader: DataLoader,
    dry_run: bool = False,
):
    eval_counter = 0
    total_loss = 0.0
    bar_len = len(dataloader) if not dry_run else 1
    with tqdm(dataloader, total=bar_len, desc="Evaluating") as pbar:
        for batch in pbar:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            targets = batch["targets"] if use_masked_targets else batch["input_ids"].detach().clone()
            loss = finetune_loss(
                outputs=outputs,
                targets=targets,
                ignore_index=data_cfg.ignore_index,
                label_smoothing=label_smoothing,
            )
            total_loss += loss
            eval_counter += 1
            pbar.update(1)
            if dry_run:
                break
    average_loss = total_loss / eval_counter
    try:
        perplexity = torch.exp(average_loss).item()
    except OverflowError:
        perplexity = float("inf")
    return perplexity


@register_task()
def main(fabric: Fabric, cfg: Dict[str, Any]):  # noqa: D103
    seed = cfg.seed
    dry_run = cfg.dry_run
    task_cfg = PerplexityConfig(**cfg.task)
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
        model.load_checkpoint(checkpoint_path, fabric=fabric, model_cfg=ckpt_model_cfg, freeze=True)
    model = fabric.setup_module(model)
    model.eval()

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

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
    result = evaluate(
        model=model,
        use_masked_targets=task_cfg.use_masked_targets,
        label_smoothing=task_cfg.label_smoothing,
        data_cfg=data_cfg,
        dataloader=dataloader,
        dry_run=dry_run,
    )
    if fabric.is_global_zero:
        eval_folder = Path(experiment_dir) / "eval"
        eval_folder.mkdir(parents=True, exist_ok=True)
        eval_file = eval_folder / f"perplexity_{data_split}.json"
        with open(eval_file, "w") as f:
            json.dump({"perplexity": result}, f)
    fabric.print(f"Perplexity on {data_cfg.dataset_name} {data_split} set: {result:.4f}")
