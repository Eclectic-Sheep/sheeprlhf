import copy
import time
from pathlib import Path
from typing import Dict

import torch
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizer

from sheeprlhf.agent.ppo import PPOAgent
from sheeprlhf.data.base import TextDataset
from sheeprlhf.data.collate import LeftPadCollate
from sheeprlhf.loss.ppo import policy_loss, value_loss
from sheeprlhf.model.actor import ActorModel
from sheeprlhf.structure.data import DataConfig
from sheeprlhf.structure.generation import GenConfig
from sheeprlhf.structure.model import ModelConfig
from sheeprlhf.structure.task import PPOConfig
from sheeprlhf.utils.data import prepare_generation_config, validate_dataset
from sheeprlhf.utils.helper import create_tensorboard_logger, get_log_dir, log_text
from sheeprlhf.utils.hydra import instantiate_from_config
from sheeprlhf.utils.metric import PPOMetricManager
from sheeprlhf.utils.model import compute_grad_norm, prepare_optimizer_parameters
from sheeprlhf.utils.ppo import AdaptiveKLController, FixedKLController, collect_rollout, masked_normalize
from sheeprlhf.utils.registry import register_task


@torch.no_grad()
def generate(  # noqa: D103
    agent: PPOAgent,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    example_prompt: Dict[str, torch.Tensor],
    device: torch.device,
):
    generated_input_ids = agent.actor.module.generate(
        input_ids=example_prompt["input_ids"].to(device),
        attention_mask=example_prompt["attention_mask"].to(device),
        generation_config=generation_config,
        use_cache=True,
    )
    prompt_length = example_prompt["input_ids"].shape[1]
    generated_attention_mask = (generated_input_ids != generation_config.pad_token_id).int()
    generated_data = {"input_ids": generated_input_ids, "attention_mask": generated_attention_mask}
    reward = agent.reward(**generated_data)[:, prompt_length:]
    action_mask = (generated_input_ids != generation_config.pad_token_id).int()[:, prompt_length:]
    last_token_idx = torch.argmax(torch.cumsum(action_mask, dim=1) * action_mask, dim=1, keepdim=True)
    reward_score = torch.gather(reward, dim=-1, index=last_token_idx).squeeze(-1)
    return tokenizer.decode(generated_input_ids[0], skip_special_tokens=True), reward_score.item()


@register_task()
def main(fabric: Fabric, cfg: Dict):  # noqa: D103
    task_cfg = PPOConfig(**cfg.task)
    model_cfg = ModelConfig(**cfg.model)
    data_cfg = DataConfig(**cfg.data)
    gen_cfg = GenConfig(**cfg.generation)
    actor_optim_cfg = cfg.actor_optimizer
    critic_optim_cfg = cfg.critic_optimizer

    fabric.seed_everything(cfg.seed + fabric.global_rank)

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger = create_tensorboard_logger(fabric, cfg, override_log_level=True)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    experiment_dir = Path(log_dir).parent

    # Setup Metrics
    metrics = PPOMetricManager(log_interval=task_cfg.log_interval).to(fabric.device)

    # Setup Dataloaders
    data_processor = validate_dataset(fabric, data_cfg)
    dataset_path = Path(data_processor.full_path)
    tokenizer = data_processor.tokenizer

    collator = LeftPadCollate(pad_value=tokenizer.pad_token_id, ignore_index=data_cfg.ignore_index)
    train_dataset = TextDataset(dataframe_path=dataset_path / "finetune_train.pkl")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=task_cfg.micro_batch_size,
        collate_fn=collator,
        num_workers=task_cfg.num_workers,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    example_prompt = torch.load(dataset_path / "example_prompt.pt")

    # Setup Model
    with fabric.init_module(empty_init=model_cfg.fabric_empty_init):
        agent = PPOAgent(model_cfg=model_cfg, task_cfg=task_cfg)
        agent.load_checkpoint(fabric=fabric)
        agent.setup_finetuning()
    agent.actor = fabric.setup_module(agent.actor)
    agent.critic = fabric.setup_module(agent.critic)
    agent.reward = fabric.setup_module(agent.reward)
    agent.reference = fabric.setup_module(agent.reference)

    # Setup Generation Configs
    generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        gen_cfg=gen_cfg,
        fabric=fabric,
    )
    eval_gen_cfg = copy.deepcopy(gen_cfg)
    eval_gen_cfg.do_sample = False
    eval_generation_config = prepare_generation_config(
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        gen_cfg=eval_gen_cfg,
        fabric=fabric,
    )

    # Setup Optimizer Scheduler fabric models

    actor_trainable_params, _, _ = prepare_optimizer_parameters(agent.actor, weight_decay=actor_optim_cfg.weight_decay)
    actor_optimizer = instantiate_from_config(
        actor_optim_cfg,
        params=actor_trainable_params,
        _convert_="partial",
    )
    actor_optimizer = fabric.setup_optimizers(actor_optimizer)

    critic_trainable_params, _, _ = prepare_optimizer_parameters(
        agent.critic, weight_decay=critic_optim_cfg.weight_decay
    )
    critic_optimizer = instantiate_from_config(
        critic_optim_cfg,
        params=critic_trainable_params,
        _convert_="partial",
    )
    critic_optimizer = fabric.setup_optimizers(critic_optimizer)

    if fabric.is_global_zero:
        gen_text, score = generate(
            agent=agent,
            tokenizer=tokenizer,
            generation_config=eval_generation_config,
            example_prompt=example_prompt,
            device=fabric.device,
        )
        log_text(fabric, gen_text, "info/example_sample", step=0)
        fabric.log("info/example_last_reward", score, step=0)

    num_training_steps = 2 if cfg.dry_run else task_cfg.epochs * len(train_dataloader)

    # KL Controller
    if task_cfg.adaptive_kl_coeff:
        kl_controller = AdaptiveKLController(
            init_kl_coef=task_cfg.init_kl_coeff, target=task_cfg.target_kl_coeff, kl_horizon=num_training_steps
        )
    else:
        kl_controller = FixedKLController(kl_coeff=task_cfg.init_kl_coeff)

    iterator = tqdm(range(num_training_steps), disable=not fabric.is_global_zero)
    data_iterator = iter(train_dataloader)
    fabric.print("Model Checkpoint interval: ", task_cfg.save_interval, "steps")
    fabric.print("Model Evaluation interval: ", task_cfg.eval_interval, "steps")
    agent.reward.eval()

    for k in iterator:
        # Setup counters and data
        if k % len(train_dataloader) == 0 or data_iterator is None:
            data_iterator = iter(train_dataloader)
        is_accumulating = (k) % task_cfg.gradient_accumulation_steps != 0
        last_step = k == num_training_steps - 1

        # Setup batch data
        batch = next(data_iterator)
        max_prompt_length = batch["prompt_input_ids"].shape[1]
        agent.actor.eval()
        agent.critic.eval()
        t0 = time.time()

        rollout, sample_output = collect_rollout(
            batch=batch,
            agent=agent,
            generation_config=generation_config,
            kl_controller=kl_controller,
            task_cfg=task_cfg,
            tokenizer=tokenizer,
            fabric=fabric,
            metrics=metrics,
        )
        time_rollout = time.time() - t0
        rollout_dataloader = DataLoader(
            rollout, batch_size=task_cfg.micro_batch_size, shuffle=True, collate_fn=lambda x: x
        )
        rollout_dataloader = fabric.setup_dataloaders(rollout_dataloader, use_distributed_sampler=False)
        agent.actor.train()
        agent.critic.train()
        for _ in range(task_cfg.ppo_epochs):
            accumulator_counter = 0
            for micro_batch in rollout_dataloader:
                is_accumulating = (accumulator_counter) % task_cfg.gradient_accumulation_steps != 0

                generated_data = {
                    "input_ids": micro_batch["input_ids"],
                    "attention_mask": micro_batch["attention_mask"],
                }
                old_log_probs = micro_batch["actor_log_probs"]
                old_values = micro_batch["values"]
                advantages = micro_batch["advantages"]
                returns = micro_batch["returns"]
                start_token_idx = max_prompt_length - 1
                action_mask = micro_batch["attention_mask"][:, start_token_idx:-1].int()
                if task_cfg.normalize_advantages:
                    advantages = masked_normalize(advantages, action_mask)

                with fabric.no_backward_sync(agent.actor, enabled=is_accumulating):
                    log_probs = agent.actor(**generated_data)[:, start_token_idx:]  # (B, num_new_tokens)
                    p_loss = policy_loss(
                        log_probs=log_probs,
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        clip_coeff=task_cfg.clip_coeff,
                        action_mask=action_mask,
                    )
                    fabric.backward(p_loss / task_cfg.gradient_accumulation_steps)

                with fabric.no_backward_sync(agent.critic, enabled=is_accumulating):
                    values = agent.critic(**generated_data)[:, start_token_idx:-1]  # (B, num_new_tokens)
                    v_loss = value_loss(
                        values=values,
                        old_values=old_values,
                        returns=returns,
                        clip_coeff=task_cfg.clip_coeff,
                        action_mask=action_mask,
                    )
                    fabric.backward((v_loss * task_cfg.vf_coeff) / task_cfg.gradient_accumulation_steps)

                if not is_accumulating:
                    actor_grads = compute_grad_norm(model=agent.actor)
                    fabric.clip_gradients(
                        agent.actor, actor_optimizer, max_norm=task_cfg.gradient_clip_val, error_if_nonfinite=True
                    )
                    actor_optimizer.step()
                    actor_optimizer.zero_grad(set_to_none=True)

                    critic_grads = compute_grad_norm(model=agent.critic)
                    fabric.clip_gradients(
                        agent.critic, critic_optimizer, max_norm=task_cfg.gradient_clip_val, error_if_nonfinite=True
                    )
                    critic_optimizer.step()
                    critic_optimizer.zero_grad(set_to_none=True)
                accumulator_counter += 1

        time_ppo = time.time() - t0 - time_rollout
        with torch.no_grad():
            metrics.info_rollout_time.update(time_rollout)
            metrics.info_ppo_time.update(time_ppo)
            metrics.train_actor_loss.update(p_loss.item())
            metrics.train_critic_loss.update(v_loss.item())
            metrics.info_actor_grad_norm.update(actor_grads)
            metrics.info_critic_grad_norm.update(critic_grads)
            metrics.info_kl_coeff.update(kl_controller.value)

        if k > 0 and (k % task_cfg.eval_interval == 0 or last_step):
            agent.actor.eval()
            agent.critic.eval()
            if fabric.is_global_zero:
                gen_text, score = generate(
                    agent=agent,
                    tokenizer=tokenizer,
                    generation_config=eval_generation_config,
                    example_prompt=example_prompt,
                    device=fabric.device,
                )
                log_text(fabric, sample_output, "info/rollout_sample", step=k)
                log_text(fabric, gen_text, "info/example_sample", step=k)
                fabric.log("info/example_last_reward", score, step=k)

        fabric.barrier()
        if k % task_cfg.log_interval == 0 or last_step:
            computed_metrics = metrics.compute_all()
            metrics.log_all(fabric=fabric, step=k, metrics_dict=computed_metrics)

            if not iterator.disable:
                description = f"iter {k}, rollout-time: {time_rollout*1000:.2f}ms, ppo-time: {time_ppo*1000:.2f}ms"
                for metric_name, metric_value in computed_metrics.items():
                    if metric_name.startswith("info/") or metric_name.startswith("debug/"):
                        continue
                    description += f", {metric_name}: {metric_value:.3f}"
                iterator.set_description(description)

        if k > 0 and (k % task_cfg.save_interval == 0 or last_step):
            checkpoint_model: ActorModel = agent.actor.module
            checkpoint_model.save_checkpoint(
                fabric=fabric,
                experiment_dir=experiment_dir,
                model_cfg=model_cfg,
                step=k,
            )
    fabric.print("Experiment output folder: ", experiment_dir)
