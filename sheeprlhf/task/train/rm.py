import time
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from sheeprlhf.data.base import TextDataset
from sheeprlhf.data.collate import CompareCollate
from sheeprlhf.loss.reward import load_reward_loss
from sheeprlhf.model.reward import RewardModel
from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.structure.task import RMConfig
from sheeprlhf.utils.data import validate_dataset
from sheeprlhf.utils.helper import create_tensorboard_logger, get_log_dir, trainable_parameter_summary
from sheeprlhf.utils.hydra import instantiate_from_config
from sheeprlhf.utils.metric import RMMetricManager, reward_accuracy
from sheeprlhf.utils.model import compute_grad_norm, get_model_checkpoint, prepare_optimizer_parameters
from sheeprlhf.utils.registry import register_task
from sheeprlhf.utils.scheduler import CosineSchedulerWithWarmup


@torch.no_grad()
def evaluate(model: RewardModel, val_dataloader: DataLoader, loss: Callable, pad_token_id: int, eval_iters: int):  # noqa: D103
    eval_counter = 0
    average_acc = 0
    average_loss = 0
    for batch in val_dataloader:
        chosen_rewards = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
        rejected_rewards = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])
        val_loss, choosen_last_rewards, rejected_last_rewards = loss(
            chosen=batch["chosen_input_ids"],
            rejected=batch["rejected_input_ids"],
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            pad_token_id=pad_token_id,
        )
        average_loss += val_loss.detach()
        acc = reward_accuracy(choosen_last_rewards, rejected_last_rewards)
        average_acc += acc
        eval_counter += 1
        if eval_iters is not None and eval_counter >= eval_iters:
            break
    average_acc /= eval_counter
    average_loss /= eval_counter
    return (
        average_loss,
        average_acc,
    )


@register_task()
def main(fabric: Fabric, cfg: Dict[str, Any]):  # noqa: D103
    task_cfg = RMConfig(**cfg.task)
    model_cfg = ModelConfig(**cfg.model)
    data_cfg = DataConfig(**cfg.data)
    optim_cfg = cfg.optim

    fabric.seed_everything(cfg.seed + fabric.global_rank)

    # Check if we are loading a finetuned model
    checkpoint_path = None
    if task_cfg.sft_experiment_dir is not None:
        ckpt_model_cfg, checkpoint_path = get_model_checkpoint(task_cfg.sft_experiment_dir, task_cfg.sft_model_name)
        fabric.print(f"Loading finetuned model from {checkpoint_path}")
    else:
        ckpt_model_cfg = model_cfg

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

    collator = CompareCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    train_dataset = TextDataset(dataframe_path=dataset_path / "reward_model_train.pkl")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=task_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=task_cfg.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    val_dataset = TextDataset(dataframe_path=dataset_path / "reward_model_validation.pkl")
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=task_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=task_cfg.num_workers,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    # Setup Model
    with fabric.init_module(empty_init=model_cfg.fabric_empty_init):
        model = RewardModel(model_cfg=ckpt_model_cfg)
        if checkpoint_path is not None:
            model.load_checkpoint(model_cfg=ckpt_model_cfg, path=checkpoint_path, device=fabric.device)
        model.setup_finetuning(model_cfg=model_cfg)

    model = fabric.setup_module(model)
    trainable_parameter_summary(model, show_names=False, fabric=fabric)

    # Setup Metrics
    metrics = RMMetricManager(log_interval=task_cfg.log_interval).to(fabric.device)

    # Setup Reward Loss
    reward_loss = load_reward_loss(task_cfg.loss_type)

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
    fabric.print("Model Checkpoint interval: ", task_cfg.save_interval, "steps")
    fabric.print("Model Evaluation interval: ", task_cfg.eval_interval, "steps")

    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)
    data_iterator = iter(train_dataloader)
    for k in iterator:
        model.train()
        # Setup counters and data
        if k % len(train_dataloader) == 0 or data_iterator is None:
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
        chosen = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        rejected = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]

        t0 = time.time()

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            chosen_rewards = model(input_ids=chosen, attention_mask=chosen_mask, use_cache=False)
            rejected_rewards = model(input_ids=rejected, attention_mask=rejected_mask, use_cache=False)
            loss, choosen_last_rewards, rejected_last_rewards = reward_loss(
                chosen=chosen,
                rejected=rejected,
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                pad_token_id=tokenizer.pad_token_id,
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
            metrics.info_choosen_reward.update(choosen_last_rewards.mean())
            metrics.info_rejected_reward.update(rejected_last_rewards.mean())

            train_acc = reward_accuracy(choosen_last_rewards, rejected_last_rewards)
            metrics.train_acc.update(train_acc)

        if k > 0 and (k % task_cfg.eval_interval == 0 or last_step):
            model.eval()
            val_loss, val_acc = evaluate(
                model=model,
                val_dataloader=val_dataloader,
                loss=reward_loss,
                pad_token_id=tokenizer.pad_token_id,
                eval_iters=task_cfg.eval_iters,
            )
            with torch.no_grad():
                metrics.val_loss.reset()
                metrics.val_acc.reset()
                metrics.val_loss.update(val_loss)
                metrics.val_acc.update(val_acc)
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
            checkpoint_model: RewardModel = model.module
            checkpoint_model.save_checkpoint(
                fabric=fabric,
                experiment_dir=experiment_dir,
                model_cfg=model_cfg,
                step=k,
            )
    fabric.print("Experiment output folder: ", experiment_dir)
