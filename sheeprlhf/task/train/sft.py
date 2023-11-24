import time
from pathlib import Path
from typing import Any, Dict

import torch
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprlhf.data.base import TextDataset
from sheeprlhf.data.collate import SFTCollate
from sheeprlhf.loss.sft import finetune_loss
from sheeprlhf.model.casual import CasualModel
from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.structure.task import SFTConfig
from sheeprlhf.utils.data import prepare_generation_config, validate_dataset
from sheeprlhf.utils.helper import (
    create_tensorboard_logger,
    get_log_dir,
    log_text,
    trainable_parameter_summary,
)
from sheeprlhf.utils.hydra import instantiate_from_config
from sheeprlhf.utils.metric import SFTMetricManager
from sheeprlhf.utils.model import compute_grad_norm, prepare_optimizer_parameters
from sheeprlhf.utils.registry import register_task
from sheeprlhf.utils.scheduler import CosineSchedulerWithWarmup


@torch.no_grad()
def evaluate(  # noqa: D103
    model: CasualModel,
    task_cfg: SFTConfig,
    data_cfg: DataConfig,
    val_dataloader: DataLoader,
) -> float:
    eval_counter = 0
    total_loss = 0.0
    eval_iters = task_cfg.eval_iters
    for batch in val_dataloader:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"] if task_cfg.use_masked_targets else batch["input_ids"].detach().clone()
        loss = finetune_loss(
            outputs=outputs,
            targets=targets,
            ignore_index=data_cfg.ignore_index,
            label_smoothing=task_cfg.label_smoothing,
        )
        total_loss += loss
        eval_counter += 1

        if eval_iters is not None and eval_counter >= eval_iters:
            break
    average_loss = total_loss / eval_counter
    return average_loss


@torch.no_grad()
def generate(  # noqa: D103
    model: CasualModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    example_prompt: Dict[str, torch.Tensor],
    device: torch.device,
) -> str:
    generated = model.generate(
        input_ids=example_prompt["input_ids"].to(device),
        attention_mask=example_prompt["attention_mask"].to(device),
        generation_config=generation_config,
    )
    generated_text = tokenizer.decode(generated[0])
    return generated_text


@register_task()
def main(fabric: Fabric, cfg: Dict[str, Any]):  # noqa: D103
    task_cfg = SFTConfig(**cfg.task)
    model_cfg = ModelConfig(**cfg.model)
    data_cfg = DataConfig(**cfg.data)
    gen_cfg = GenConfig(**cfg.generation)
    optim_cfg = cfg.optim

    fabric.seed_everything(cfg.seed + fabric.global_rank)

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger = create_tensorboard_logger(fabric, cfg, override_log_level=True)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    experiment_dir = Path(log_dir).parent

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

    collator = SFTCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    train_dataset = TextDataset(dataframe_path=dataset_path / "finetune_train.pkl")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=task_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=task_cfg.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    val_dataset = TextDataset(dataframe_path=dataset_path / "finetune_validation.pkl")
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=task_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=task_cfg.num_workers,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    example_prompt = torch.load(dataset_path / "example_prompt.pt")

    # Setup Model
    with fabric.init_module(empty_init=model_cfg.fabric_empty_init):
        model = CasualModel(model_cfg=model_cfg)
        model.setup_finetuning(fabric)
    model = fabric.setup_module(model)
    trainable_parameter_summary(model, show_names=False, fabric=fabric)

    # Setup Generation Config
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        gen_cfg=gen_cfg,
        fabric=fabric,
    )

    # Setup Metrics
    metrics = SFTMetricManager(log_interval=task_cfg.log_interval).to(fabric.device)

    # Setup Optimizer Scheduler
    trainable_params, _, _ = prepare_optimizer_parameters(model, optim_cfg.weight_decay)
    optimizer = instantiate_from_config(
        optim_cfg,
        params=trainable_params,
        _convert_="partial",
    )
    num_training_steps = 2 if cfg.dry_run else task_cfg.epochs * len(train_dataloader)
    lr_scheduler = CosineSchedulerWithWarmup(
        lr=optim_cfg.lr,
        warmup_steps=task_cfg.lr_warmup_steps,
        lr_decay_steps=num_training_steps,
    )
    optimizer = fabric.setup_optimizers(optimizer)
    model.eval()
    gen_text = generate(
        model=model.module,
        tokenizer=tokenizer,
        generation_config=generation_config,
        example_prompt=example_prompt,
        device=fabric.device,
    )
    log_text(fabric, gen_text, "info/example_sample", step=0)
    fabric.print("Model Checkpoint interval: ", task_cfg.save_interval, "steps")
    fabric.print("Model Evaluation interval: ", task_cfg.eval_interval, "steps")

    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)

    data_iterator = iter(train_dataloader)
    for k in iterator:
        model.train()
        # Setup counters and data
        if k % len(train_dataloader) == 0:
            data_iterator = iter(train_dataloader)
        is_accumulating = (k) % task_cfg.gradient_accumulation_steps != 0
        last_step = k == num_training_steps - 1

        # Setup learning rate
        lr = lr_scheduler.get_lr(it=k)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        metrics.info_lr.update(lr)

        # Setup batch data
        batch = next(data_iterator)
        input_ids = batch["input_ids"]  # type: ignore[index]
        attention_mask = batch["attention_mask"]  # type: ignore[index]
        targets = batch["targets"] if task_cfg.use_masked_targets else input_ids.detach().clone()  # type: ignore[index]

        num_tokens = input_ids.numel()
        padding_pct = 100 * (attention_mask == 0).sum().item() / num_tokens

        # Forward and Backward Pass
        t0 = time.time()
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            loss = finetune_loss(
                outputs=outputs,
                targets=targets,
                ignore_index=data_cfg.ignore_index,
                label_smoothing=task_cfg.label_smoothing,
            )
            fabric.backward(loss / task_cfg.gradient_accumulation_steps)

        dt = time.time() - t0
        if not is_accumulating:
            metrics.info_grad_norm.update(compute_grad_norm(model))
            fabric.clip_gradients(model, optimizer, max_norm=task_cfg.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            metrics.info_time.update(dt)
            metrics.train_loss.update(loss.item())
            metrics.info_tokens_per_seconds.update(num_tokens / dt)
            metrics.info_padding_percentage.update(padding_pct)

        if k > 0 and (k % task_cfg.eval_interval == 0 or last_step):
            model.eval()
            val_loss = evaluate(
                model=model,
                task_cfg=task_cfg,
                data_cfg=data_cfg,
                val_dataloader=val_dataloader,
            )
            # we don't want to take average of different val losses
            # we already computed average inside evaluate function
            with torch.no_grad():
                metrics.val_loss.reset()
                metrics.val_loss.update(val_loss)

            if fabric.is_global_zero:
                gen_text = generate(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    example_prompt=example_prompt,
                    device=fabric.device,
                )
                log_text(fabric, gen_text, "info/example_sample", step=k)
        fabric.barrier()
        if k > 0 and (k % task_cfg.log_interval == 0 or last_step):
            computed_metrics = metrics.compute_all()
            metrics.log_all(fabric=fabric, step=k, metrics_dict=computed_metrics)

            if not iterator.disable:
                description = f"iter {k}, time: {dt*1000:.2f}ms"
                for metric_name, metric_value in computed_metrics.items():
                    if metric_name.startswith("info/"):
                        continue
                    description += f", {metric_name}: {metric_value:.3f}"
                iterator.set_description(description)
        if k > 0 and (k % task_cfg.save_interval == 0 or last_step):
            checkpoint_model: CasualModel = model.module
            checkpoint_model.save_checkpoint(
                fabric=fabric,
                experiment_dir=experiment_dir,
                model_cfg=model_cfg,
                step=k,
            )
    fabric.print("Experiment output folder: ", experiment_dir)
