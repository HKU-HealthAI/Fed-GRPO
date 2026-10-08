"""
GRPO Client: Client-side local training logic using TRL's GRPOTrainer
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Callable, Any, Optional, Tuple, Union
import logging
import copy
import os
import gc

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from utils.data import PromptDataset
from utils.experiment import is_main_process


class WeightedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass that supports per-prompt loss weighting based on global reward calibration.

    When global_reward_mean/std are provided, each prompt's loss contribution is weighted by:
        ω_q = 1 + λ · tanh((μ_global - μ_local) / (σ_global + ε))
    where μ_local is the mean reward of the prompt's generation group.
    """

    def __init__(self, *args, global_reward_mean: Optional[float] = None,
                 global_reward_std: Optional[float] = None,
                 calibration_lambda: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_reward_mean = global_reward_mean
        self.global_reward_std = global_reward_std
        self.calibration_lambda = calibration_lambda
        self._current_prompt_weights = None  # Set during _generate_and_score_completions

    def _generate_and_score_completions(self, inputs):
        first_reward_name = self.reward_func_names[0]

        output = super()._generate_and_score_completions(inputs)

        # Compute per-prompt weights if global stats are available
        if self.global_reward_mean is not None and self.global_reward_std is not None:
            # TRL stores rewards in deque(maxlen=generation_batch_size), so use current buffer directly.
            new_rewards = list(self._logs["rewards"][first_reward_name])
            if len(new_rewards) == 0:
                return output

            # Safety: ensure rewards can be grouped by num_generations.
            if len(new_rewards) % self.num_generations != 0:
                valid_len = (len(new_rewards) // self.num_generations) * self.num_generations
                if valid_len == 0:
                    return output
                new_rewards = new_rewards[-valid_len:]

            device = output["advantages"].device
            rewards_tensor = torch.tensor(new_rewards, dtype=torch.float32, device=device)

            # Per-prompt mean reward: reshape (total_completions,) -> (num_prompts, num_generations)
            per_prompt_mean = rewards_tensor.view(-1, self.num_generations).mean(dim=1)

            # ω_q = 1 + λ · tanh((μ_global - μ_local) / (σ_global + ε))
            sigma = self.global_reward_std if self.global_reward_std > 1e-8 else 1e-8
            raw = (self.global_reward_mean - per_prompt_mean) / (sigma + 1e-8)
            prompt_weights = 1.0 + self.calibration_lambda * torch.tanh(raw)

            # Expand to completion level and slice to local process
            prompt_weights = prompt_weights.repeat_interleave(self.num_generations, dim=0)
            process_slice = slice(
                self.accelerator.process_index * len(output["advantages"]),
                (self.accelerator.process_index + 1) * len(output["advantages"]),
            )
            prompt_weights = prompt_weights[process_slice]
            if prompt_weights.numel() == 0:
                return output

            # Log weight stats
            _logger = logging.getLogger(__name__)
            if is_main_process():
                _logger.info(
                    f"Global reward calibration: μ_global={self.global_reward_mean:.4f}, "
                    f"σ_global={self.global_reward_std:.4f}, "
                    f"ω_q range=[{prompt_weights.min().item():.4f}, {prompt_weights.max().item():.4f}], "
                    f"ω_q mean={prompt_weights.mean().item():.4f}"
                )

            output["prompt_weights"] = prompt_weights
        else:
            self._current_prompt_weights = None

        return output

    def _compute_loss(self, model, inputs):
        prompt_weights = inputs.get("prompt_weights", None)

        # If no prompt weights, fall back to standard GRPO loss
        if prompt_weights is None:
            return super()._compute_loss(model, inputs)

        # --- Replicate parent _compute_loss with weighted aggregation for "grpo" loss_type ---
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # --- Weighted loss aggregation (only difference from parent) ---
        if self.loss_type == "grpo":
            per_prompt_loss = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            # Weighted mean instead of simple .mean()
            loss = (per_prompt_loss * prompt_weights).sum() / prompt_weights.sum()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics (same as parent)
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        from trl.trainer.grpo_trainer import nanmin, nanmax
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss


class StepLoggingCallback(TrainerCallback):
    """Prepend step number and write TRL metrics to training.log via Python logging."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            saved = dict(logs)
            logs.clear()
            logs.update({"step": state.global_step, **saved})
            # Also write key metrics to training.log through the logging module
            _logger = logging.getLogger(__name__)
            if is_main_process() and 'loss' in logs:
                # Pick the most important metrics for a concise log line
                step = state.global_step
                loss = logs.get('loss', 'N/A')
                reward = logs.get('reward', 'N/A')
                reward_std = logs.get('reward_std', 'N/A')
                frac_zero_std = logs.get('frac_reward_zero_std', 'N/A')
                grad_norm = logs.get('grad_norm', 'N/A')
                lr = logs.get('learning_rate', 'N/A')
                mean_len = logs.get('completions/mean_length', 'N/A')
                clipped = logs.get('completions/clipped_ratio', 'N/A')
                entropy = logs.get('entropy', 'N/A')
                _logger.info(
                    f"[Step {step}] loss={loss}, reward={reward}, reward_std={reward_std}, "
                    f"frac_zero_std={frac_zero_std}, grad_norm={grad_norm}, lr={lr}, "
                    f"mean_len={mean_len}, clipped_ratio={clipped}, entropy={entropy}"
                )


class CheckpointCallback(TrainerCallback):
    """Save checkpoint every N steps during training (for centralized/local_only modes)."""
    
    def __init__(self, save_steps_freq: int, checkpoint_base_dir: str, tokenizer, client_name: str = ""):
        """
        Args:
            save_steps_freq: Save checkpoint every N steps (0 = disabled)
            checkpoint_base_dir: Base directory for saving checkpoints
            tokenizer: Tokenizer to save with model
            client_name: Optional client name for checkpoint naming
        """
        self.save_steps_freq = save_steps_freq
        self.checkpoint_base_dir = checkpoint_base_dir
        self.tokenizer = tokenizer
        self.client_name = client_name
        self._logger = logging.getLogger(__name__)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.save_steps_freq <= 0:
            return
        
        # Check if it's time to save
        if state.global_step > 0 and state.global_step % self.save_steps_freq == 0:
            self._save_checkpoint(model, state.global_step)
    
    def _save_checkpoint(self, model, step: int):
        """Save model checkpoint at current step."""
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            
            # Unwrap model from Accelerate wrapper
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Synchronize all processes before saving
            accelerator.wait_for_everyone()
            
            # Only main process saves
            if accelerator.is_main_process:
                checkpoint_dir = os.path.join(
                    self.checkpoint_base_dir, 
                    f"checkpoint_step_{step}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                self._logger.info(f"Saving checkpoint at step {step} to {checkpoint_dir}")
                unwrapped_model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Wait for saving to complete
            accelerator.wait_for_everyone()
        except ImportError:
            # Fallback without Accelerator
            if is_main_process():
                checkpoint_dir = os.path.join(
                    self.checkpoint_base_dir,
                    f"checkpoint_step_{step}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                self._logger.info(f"Saving checkpoint at step {step} to {checkpoint_dir}")
                model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)

logger = logging.getLogger(__name__)


# Reward function logging interval (log sample every N completions)
REWARD_LOG_INTERVAL = 50


class GRPOClient:
    """Unified GRPO Client using TRL's GRPOTrainer"""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        tokenizer: Any,
        train_data: List[Dict[str, Any]],
        reward_func: Optional[Callable] = None,
        reward_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        federation_round: int = 0
    ):
        """
        Initialize GRPO Client
        
        Args:
            client_id: Client ID (use 0 for centralized training)
            model: Local model
            tokenizer: Tokenizer
            train_data: Local training data
            reward_func: Reward function (priority)
            reward_model: Reward model object
            config: Training configuration
            device: Device
            federation_round: Current federation round
        """
        self.client_id = client_id
        # Use Accelerator's device if available, otherwise use specified device
        try:
            from accelerate import PartialState
            self.device = PartialState().device
        except (ImportError, RuntimeError) as e:
            if is_main_process():
                logger.debug(f"PartialState not available, using specified device: {e}")
            self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.config = config or {}
        
        # Reward function (prefer reward_func, otherwise use reward_model)
        if reward_func is not None:
            self._base_reward_func = reward_func
        elif reward_model is not None:
            self._base_reward_func = lambda response, reference: reward_model.compute_rewards([response], [reference])[0]
        else:
            raise ValueError("Must provide either reward_func or reward_model")
        
        # Reference model (for KL divergence)
        self.ref_model: Optional[nn.Module] = None
        
        # GRPOTrainer instance (created during training)
        self.trainer: Optional[GRPOTrainer] = None
        
        # Federation round
        self.federation_round = federation_round

        # Global reward calibration stats (set externally via set_global_reward_stats)
        self.global_reward_mean = None
        self.global_reward_std = None

        # Round counter for varying random seed each round (incremented in local_train)
        self.train_round = 0
        
        if is_main_process():
            logger.info(f"Client {client_id} initialized with {len(train_data)} samples (using TRL GRPOTrainer)")

    def set_global_reward_stats(self, mean: float, std: float):
        """Set global reward statistics for per-prompt loss calibration."""
        self.global_reward_mean = mean
        self.global_reward_std = std

    def _create_reward_function(self) -> Callable[[List[str], List[str]], List[float]]:
        """Create reward function compatible with TRL's GRPOTrainer
                
        Returns:
            Callable reward function for GRPOTrainer
        """
        base_reward_func = self._base_reward_func
        client_id = self.client_id
        
        # Counter for logging samples periodically
        call_counter = [0]  # Use list to allow modification in nested function

        def reward_func(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
            """
            Reward function for GRPOTrainer

            Args:
                completions: List of generated completions
                prompts: List of prompts
                **kwargs: Additional columns from dataset (answer, etc.)

            Returns:
                List of reward values
            """
            # Extract ground truth from kwargs (passed by TRL from dataset columns)
            answers = kwargs.get('answer', kwargs.get('reference', [None] * len(completions)))
            
            # Ensure answers list has the same length as completions
            if len(answers) != len(completions):
                if is_main_process():
                    logger.warning(
                        f"[Client {client_id}] Mismatch: {len(answers)} answers for {len(completions)} completions. "
                        f"Padding with empty strings."
                    )
                # Pad or truncate answers to match completions length
                if len(answers) < len(completions):
                    answers = answers + [''] * (len(completions) - len(answers))
                else:
                    answers = answers[:len(completions)]
            
            rewards = []
            original_completion_lengths = []  # Track lengths of original completions for logging
            
            for idx, completion in enumerate(completions):
                call_counter[0] += 1

                # Log sample completion periodically for debugging
                should_log = (call_counter[0] % REWARD_LOG_INTERVAL == 1)
                
                # Track original completion length before post-processing
                original_len = len(completion)
                original_completion_lengths.append(original_len)
                
                # Get ground truth answer for this sample (handle None case)
                ground_truth = answers[idx] if answers[idx] is not None else ''
                
                # Compute reward
                try:
                    reward = base_reward_func(completion, ground_truth)
                    # Ensure reward is a valid number
                    if not isinstance(reward, (int, float)) or reward != reward:  # NaN check
                        if is_main_process():
                            logger.warning(f"Invalid reward value: {reward}, using 0.0")
                        reward = 0.0
                except Exception as e:
                    if is_main_process():
                        logger.warning(f"Reward computation failed: {e}")
                    reward = 0.0

                # Log sample completion periodically
                if should_log and is_main_process():
                    completion_len = len(completion)
                    # Truncate completion for logging (show first 200 and last 200 chars)
                    if len(completion) > 500:
                        completion_preview = completion[:200] + "\n... [truncated] ...\n" + completion[-200:]
                    else:
                        completion_preview = completion
                    logger.info(
                        f"[Client {client_id}] Sample #{call_counter[0]}: "
                        f"len={completion_len}, reward={reward:.4f}, "
                        f"ground_truth={str(ground_truth)[:50]},"
                        f"completion_preview={completion_preview}"
                    )
                
                # Log completion preview when reward is 0 (for debugging reward collapse)
                if reward == 0.0 and is_main_process():
                    # Truncate completion for logging (show first 200 and last 200 chars)
                    if len(completion) > 500:
                        completion_preview = completion[:200] + "\n... [truncated] ...\n" + completion[-200:]
                    else:
                        completion_preview = completion
                    logger.debug(
                        f"[Client {client_id}] Zero reward - Sample #{call_counter[0]}: "
                        f"Completion preview:\n{completion_preview}\n"
                        f"ground_truth={str(ground_truth)[:50]}"
                    )

                rewards.append(float(reward))

            return rewards
        
        return reward_func
    
    def _create_grpo_config(
        self,
        output_dir: str,
        max_steps: int = 10,
        logging_steps: int = 1,
        is_federated: bool = False
    ) -> GRPOConfig:
        """Create GRPOConfig with all training settings.
        
        Args:
            output_dir: Output directory for logs and checkpoints
            max_steps: Maximum training steps
            logging_steps: Logging interval
            is_federated: If True, disable TRL's sync_ref_model
            
        Returns:
            Configured GRPOConfig instance
        """
        training_config = self.config.get('training', {})
        
        config_kwargs = {
            # Basic training settings
            "output_dir": output_dir,
            "per_device_train_batch_size": training_config.get('per_device_train_batch_size', 4),
            "gradient_accumulation_steps": training_config.get('gradient_accumulation_steps', 2),
            "learning_rate": training_config.get('learning_rate', 1.0e-6),
            "max_steps": max_steps,
            "num_train_epochs": training_config.get('num_train_epochs', 1),
            
            # GRPO specific parameters
            "num_generations": training_config.get('num_generations', 16),
            "beta": training_config.get('beta_kl', 0),
            "max_completion_length": training_config.get('max_completion_length', 512),
            "max_prompt_length": training_config.get('max_prompt_length', 256),
            "epsilon": training_config.get('epsilon', 0.2),
            "epsilon_high": training_config.get('epsilon_high', None),
            "num_iterations": training_config.get('num_iterations', 1),
            "loss_type": training_config.get('loss_type', 'grpo'),
            "mask_truncated_completions": training_config.get('mask_truncated_completions', False),

            # Optimizer settings
            "optim": training_config.get('optim', 'adamw_torch_fused'),
            "weight_decay": training_config.get('weight_decay', 0.0),
            "adam_beta1": training_config.get('adam_beta1', 0.9),
            "adam_beta2": training_config.get('adam_beta2', 0.999),
            "max_grad_norm": training_config.get('max_grad_norm', 1.0),
            "lr_scheduler_type": training_config.get('lr_scheduler_type', 'constant'),
            "warmup_steps": training_config.get('warmup_steps', 0),
            "warmup_ratio": training_config.get('warmup_ratio', 0.0),
            "bf16": training_config.get('bf16', True),

            # Generation settings
            "temperature": training_config.get('temperature', 1.0),
            "top_p": training_config.get('top_p', 1.0),

            # vLLM settings
            "use_vllm": training_config.get('use_vllm', False),
            "vllm_model_impl": "transformers",
            "vllm_server_port": training_config.get('vllm_server_port', 8000),
            "vllm_importance_sampling_correction": training_config.get(
                'vllm_importance_sampling_correction', training_config.get('beta_kl', 0) > 0
            ),
            "cast_lm_head_to_fp32": True,

            # Reference model settings (federated: server controls; centralized: TRL controls)
            "sync_ref_model": (training_config.get('sync_ref_model', False)) and (not is_federated),
            "ref_model_sync_steps": training_config.get('ref_model_update_interval', 100),
            "ref_model_mixup_alpha": training_config.get('ref_model_mixup_alpha', 1.0),

            # Logging settings
            "logging_steps": logging_steps,
            "logging_dir": f"{output_dir}/logs",
            "report_to": "none",
            "save_strategy": "no",
            
            # Other settings
            "remove_unused_columns": False,
            "seed": 42 + self.client_id * 1000 + self.train_round,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "ddp_find_unused_parameters": False,
        }
        
        # Add stop tokens for R1-style format
        use_vllm = training_config.get('use_vllm', False)
        if use_vllm:
            config_kwargs["generation_kwargs"] = {
                "stop": ["</answer>"],  # vLLM uses "stop"
                "include_stop_str_in_output": True,  # Include stop string in output
            }
        else:
            config_kwargs["generation_kwargs"] = {
                "stop_strings": ["</answer>"],
            }        

        return GRPOConfig(**config_kwargs)
    
    def _create_trainer(
        self,
        grpo_config: GRPOConfig,
        train_dataset: PromptDataset,
        reward_func: Callable
    ) -> GRPOTrainer:
        """Create and return a GRPOTrainer instance.
        
        Args:
            grpo_config: GRPO configuration
            train_dataset: Training dataset
            reward_func: Reward function
            
        Returns:
            Configured GRPOTrainer instance
        """
        trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            reward_funcs=reward_func,
        )

        # Check if we should use WeightedGRPOTrainer for global reward calibration
        federated_config = self.config.get('federated', {})
        use_calibration = federated_config.get('use_global_reward_calibration', False)
        if use_calibration and self.global_reward_mean is not None and self.global_reward_std is not None:
            calibration_lambda = federated_config.get('global_reward_calibration_lambda', 0.5)
            trainer = WeightedGRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
                reward_funcs=reward_func,
                global_reward_mean=self.global_reward_mean,
                global_reward_std=self.global_reward_std,
                calibration_lambda=calibration_lambda,
            )
            if is_main_process():
                logger.info(
                    f"Client {self.client_id}: using WeightedGRPOTrainer "
                    f"(λ={calibration_lambda}, μ_global={self.global_reward_mean:.4f}, "
                    f"σ_global={self.global_reward_std:.4f})"
                )
        
        training_config = self.config.get('training', {})
        use_vllm = training_config.get('use_vllm', False)
        
        # Fix EOS token handling for models where pad_token_id differs from eos_token_id
        if not use_vllm and self.tokenizer is not None:
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None and eos_id is not None and pad_id != eos_id:
                # 1. Make generate() stop at both <|im_end|> and <|endoftext|>
                trainer.generation_config.eos_token_id = [eos_id, pad_id]
                
                # 2. Patch _generate_single_turn to also truncate completions at pad_token_id
                _orig_gen_single = trainer._generate_single_turn
                _extra_eos_id = pad_id
                
                def _eos_patched_gen_single(prompts):
                    prompt_ids, comp_ids, logprobs, extra = _orig_gen_single(prompts)
                    fixed = []
                    for ids in comp_ids:
                        try:
                            idx = ids.index(_extra_eos_id)
                            fixed.append(ids[:idx + 1])  # include stop token (consistent with TRL)
                        except ValueError:
                            fixed.append(ids)
                    return prompt_ids, fixed, logprobs, extra
                
                trainer._generate_single_turn = _eos_patched_gen_single
                
                if is_main_process():
                    logger.info(
                        f"Fixed EOS handling: generation stops at eos_token_id={[eos_id, pad_id]}, "
                        f"completions also truncated at pad_token_id={pad_id}"
                    )

        # Bind tokenizer to generation_config for non-vLLM mode
        if not use_vllm and self.tokenizer is not None:
            # Get the unwrapped model to avoid recursive patching through Accelerate wrappers
            unwrapped_model = trainer.model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            
            # Check if already patched by looking for our marker
            if not hasattr(unwrapped_model, '_tokenizer_inject_patched'):
                # Get the true original generate from the unwrapped model class
                original_generate = unwrapped_model.generate.__func__ if hasattr(unwrapped_model.generate, '__func__') else unwrapped_model.generate
                tokenizer = self.tokenizer

                # Define a new wrapper function
                def generate_with_tokenizer(self_model, *args, **kwargs):
                    # If tokenizer is not in kwargs, inject it manually
                    if "tokenizer" not in kwargs:
                        kwargs["tokenizer"] = tokenizer
                    return original_generate(self_model, *args, **kwargs)

                # Bind as a method and replace
                import types
                unwrapped_model.generate = types.MethodType(generate_with_tokenizer, unwrapped_model)
                unwrapped_model._tokenizer_inject_patched = True
                
                if is_main_process():
                    logger.info("Patched model.generate to inject tokenizer for stop_strings support.")
        
        trainer.callback_handler.callbacks.insert(0, StepLoggingCallback())

        return trainer

    def train(
        self,
        client_name: Optional[str] = None,
        metrics_logger: Optional[Any] = None,
        checkpoint_base_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train using GRPOTrainer (step-based, standard GRPO training)

        Used for local_only and centralized training modes.

        Args:
            client_name: Custom client name (for logging)
            metrics_logger: Optional TrainingMetricsLogger for per-step logging
            checkpoint_base_dir: Base directory for saving periodic checkpoints

        Returns:
            Training metrics dictionary
        """
        name = client_name if client_name else f"Client {self.client_id}"
        
        # Create temporary output directory
        output_dir = self._create_temp_output_dir(name.replace(' ', '_'))
        
        # Read config parameters
        training_config = self.config.get('training', {})
        max_steps = training_config.get('max_steps', 500)
        use_vllm = training_config.get('use_vllm', False)
        save_steps_freq = training_config.get('save_steps_freq', 0)

        if is_main_process():
            logger.info(f"{name} starting training")

        # Create GRPO config for step-based training (is_federated=False for centralized/local_only)
        grpo_config = self._create_grpo_config(
            output_dir=output_dir,
            max_steps=max_steps,
            logging_steps=1,
            is_federated=False
        )
        
        # Create dataset, reward function, and trainer
        train_dataset = PromptDataset(self.train_data, self.tokenizer)
        reward_func = self._create_reward_function()
        trainer = self._create_trainer(grpo_config, train_dataset, reward_func)
        
        # Add checkpoint callback if save_steps_freq > 0 and checkpoint_base_dir is provided
        if save_steps_freq > 0 and checkpoint_base_dir:
            checkpoint_callback = CheckpointCallback(
                save_steps_freq=save_steps_freq,
                checkpoint_base_dir=checkpoint_base_dir,
                tokenizer=self.tokenizer,
                client_name=name
            )
            trainer.add_callback(checkpoint_callback)
            if is_main_process():
                logger.info(f"Checkpoint saving enabled: every {save_steps_freq} steps to {checkpoint_base_dir}")
        
        try:
            # Train
            train_result = trainer.train()

            # Log per-step metrics to metrics_logger (for centralized/local_only training)
            if metrics_logger:
                log_history = trainer.state.log_history
                train_step_metrics = [
                    entry for entry in log_history
                    if 'loss' in entry and not any(k.startswith('eval_') for k in entry.keys())
                ]
                for step_idx, step_data in enumerate(train_step_metrics):
                    step_metrics = {
                        'loss': step_data.get('loss', 0.0),
                        'avg_reward': step_data.get('reward', 0.0),
                        'reward_std': step_data.get('reward_std', 0.0),
                        'avg_response_length': step_data.get('completions/mean_length', 0.0),
                        'avg_grad_norm': step_data.get('grad_norm', 0.0),
                        'avg_kl': step_data.get('kl', 0.0),
                    }
                    metrics_logger.log_step(step_idx + 1, step_metrics, client_name=name)

            # Extract metrics using shared method
            metrics = self._extract_training_metrics(trainer, train_result, max_steps)
            # Add num_valid_samples for backward compatibility
            metrics['num_valid_samples'] = metrics['num_samples']

            if is_main_process():
                logger.info(
                    f"{name} training completed: "
                    f"steps={max_steps}, "
                    f"loss={metrics['loss']:.4f}, "
                    f"reward={metrics['avg_reward']:.4f}, "
                    f"response_length={metrics['avg_response_length']:.1f}, "
                    f"grad_norm={metrics['avg_grad_norm']:.4f}"
                )
            
            # Update model reference
            if self.model is not trainer.model:
                del self.model
            self.model = trainer.model
            
            return metrics
        
        finally:
            # Clean up temporary output directory (main process only in distributed)
            try:
                from accelerate import PartialState
                if PartialState().is_main_process:
                    self._cleanup_after_training(output_dir, name)
            except (ImportError, RuntimeError):
                self._cleanup_after_training(output_dir, name)
            self._cleanup_vllm_resources(trainer, use_vllm, name)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if is_main_process():
                    logger.debug(f"{name}: cleared CUDA cache in finally block")

    def local_train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Local GRPO training for federated rounds.
        
        Returns:
            Tuple of (updated_weights, metrics)
        """
        # Read training parameters from config
        training_config = self.config.get('training', {})
        federated_config = self.config.get('federated', {})
        
        max_steps = federated_config.get('local_steps', 10)
        use_vllm = training_config.get('use_vllm', False)
                        
        if is_main_process():
            logger.info(f"Client {self.client_id} local training")
        
        self.train_round += 1
        name = f"Client {self.client_id}"
        # Use federation_round for unique temp directory naming across federated rounds
        output_dir = self._create_temp_output_dir(f"client_{self.client_id}", self.federation_round)
        
        grpo_config = self._create_grpo_config(
            output_dir=output_dir,
            max_steps=max_steps,
            logging_steps=1,
            is_federated=True
        )
        
        train_dataset = PromptDataset(self.train_data, self.tokenizer)
        reward_func = self._create_reward_function()
        self.trainer = self._create_trainer(grpo_config, train_dataset, reward_func)
        
        if training_config.get('beta_kl', 0) > 0 and self.ref_model is not None and hasattr(self.trainer, 'ref_model') and self.trainer.ref_model is not None:
            if is_main_process():
                logger.info(f"{name}: Loading server ref_model weights (per-round sync)")
            ref_model_to_load = self.trainer.ref_model
            if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
                ref_model_to_load = self.trainer.accelerator.unwrap_model(self.trainer.ref_model)
            ref_model_to_load.load_state_dict(self.ref_model.state_dict())
        
        if is_main_process():
            logger.info(f"{name} starting GRPOTrainer training...")
        
        try:
            train_result = self.trainer.train()
            metrics = self._extract_training_metrics(self.trainer, train_result, max_steps)
            if is_main_process():
                logger.info(
                    f"{name} completed training: "
                    f"loss={metrics['loss']:.4f}, "
                    f"reward={metrics['avg_reward']:.4f}, "
                    f"response_length={metrics['avg_response_length']:.1f}, "
                    f"grad_norm={metrics['avg_grad_norm']:.4f}"
                )

            # Unwrap model from DDP/Accelerate to get clean model without stale hooks
            if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
                trained_model = self.trainer.accelerator.unwrap_model(self.trainer.model)
            else:
                trained_model = self.trainer.model
            if self.model is not trained_model:
                del self.model
            self.model = trained_model

            return {k: v.cpu() for k, v in self.model.state_dict().items()}, metrics

        finally:
            # Free accelerator/DDP resources to prevent NCCL state leaks
            if self.trainer is not None and hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
                self.trainer.accelerator.free_memory()
            # Clean up temporary output directory (main process only in distributed)
            try:
                from accelerate import PartialState
                if PartialState().is_main_process:
                    self._cleanup_after_training(output_dir, name)
            except (ImportError, RuntimeError):
                self._cleanup_after_training(output_dir, name)
            self._cleanup_vllm_resources(self.trainer, use_vllm, name)
            self.trainer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_ref_weights(self, ref_weights: Dict[str, torch.Tensor]):
        """Load reference model weights for KL divergence computation"""
        if self.ref_model is None:
            self.ref_model = copy.deepcopy(self.model)
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        self.ref_model.load_state_dict({k: v.to(self.device) for k, v in ref_weights.items()})
    
    def _extract_training_metrics(
        self, 
        trainer: GRPOTrainer, 
        train_result: Any,
        max_steps: int
    ) -> Dict[str, float]:
        """Extract and compute training metrics from trainer's log history
        
        Args:
            trainer: The GRPOTrainer instance
            train_result: Result from trainer.train()
            max_steps: Maximum training steps
            
        Returns:
            Dictionary of computed metrics
        """
        log_history = trainer.state.log_history
        
        # Collect all training step metrics (exclude eval)
        train_step_metrics = [
            entry for entry in log_history
            if 'loss' in entry and not any(k.startswith('eval_') for k in entry.keys())
        ]
        
        # Compute average metrics across all steps
        if train_step_metrics:
            n = len(train_step_metrics)
            avg_reward_std = sum(m.get('reward_std', 0.0) for m in train_step_metrics) / n
            avg_reward = sum(m.get('reward', 0.0) for m in train_step_metrics) / n
            avg_loss = sum(m.get('loss', 0.0) for m in train_step_metrics) / n
            avg_kl = sum(m.get('kl', 0.0) for m in train_step_metrics) / n
            avg_response_length = sum(m.get('completions/mean_length', 0.0) for m in train_step_metrics) / n
            avg_frac_zero_std = sum(m.get('frac_reward_zero_std', 0.0) for m in train_step_metrics) / n
            grad_norms = [m.get('grad_norm', 0.0) for m in train_step_metrics if 'grad_norm' in m]
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            max_grad_norm = max(grad_norms) if grad_norms else 0.0
        else:
            # Fallback to final metrics
            avg_reward_std = train_result.metrics.get('train_reward_std', 0.0)
            avg_reward = train_result.metrics.get('train_reward', 0.0)
            avg_loss = train_result.metrics.get('train_loss', 0.0)
            avg_kl = 0.0
            avg_response_length = train_result.metrics.get('train_completions/mean_length', 0.0)
            avg_frac_zero_std = 0.0
            avg_grad_norm = 0.0
            max_grad_norm = 0.0

        return {
            'num_samples': len(self.train_data),
            'num_valid_steps': max_steps,
            'loss': avg_loss,
            'avg_reward': avg_reward,
            'avg_kl': avg_kl,
            'reward_std': avg_reward_std,
            'frac_reward_zero_std': avg_frac_zero_std,
            'avg_response_length': avg_response_length,
            'avg_grad_norm': avg_grad_norm,
            'max_grad_norm': max_grad_norm,
        }

    def _cleanup_vllm_resources(self, trainer: GRPOTrainer, use_vllm: bool, name: str) -> None:
        """Clean up vLLM resources after training
        
        Args:
            trainer: The GRPOTrainer instance
            use_vllm: Whether vLLM was used
            name: Name for logging (e.g., "Client 0")
        """
        if not use_vllm or trainer is None:
            return
            
        # Clean up vLLM client communicator (server mode)
        if getattr(trainer, "vllm_client", None) is not None:
            vc = trainer.vllm_client
            if hasattr(vc, "close_communicator"):
                try:
                    vc.close_communicator()
                    if is_main_process():
                        logger.info(f"{name}: closed vLLM communicator")
                except Exception as e:
                    if is_main_process():
                        logger.warning(f"{name}: failed to close vLLM communicator: {e}")
        
        # Clean up colocated vLLM LLM object (colocate mode)
        if getattr(trainer, "llm", None) is not None:
            try:
                del trainer.llm
                trainer.llm = None
                if is_main_process():
                    logger.info(f"{name}: released colocated vLLM LLM object")
            except Exception as e:
                if is_main_process():
                    logger.warning(f"{name}: failed to release vLLM LLM object: {e}")

    def _cleanup_after_training(self, output_dir: str, name: str) -> None:
        """Clean up temporary directory and CUDA cache after training
        
        Args:
            output_dir: Temporary output directory to remove
            name: Name for logging
        """
        import shutil
        try:
            shutil.rmtree(output_dir)
            if is_main_process():
                logger.info(f"Cleaned up temporary directory: {output_dir}")
        except Exception as e:
            if is_main_process():
                logger.warning(f"Failed to cleanup temporary directory {output_dir}: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if is_main_process():
                logger.debug(f"{name}: cleared CUDA cache")

    def _create_temp_output_dir(self, name: str, round_num: Optional[int] = None) -> str:
        """Create temporary output directory for training.

        Args:
            name: Name identifier (e.g., "client_0" or "Centralized_Server")
            round_num: Optional round number for federated training

        Returns:
            Path to created temporary directory
        """
        # Use experiment-specific log_dir to avoid collisions between concurrent jobs
        log_dir = self.config.get('_log_dir', None)
        if log_dir:
            tmp_base = os.path.join(log_dir, 'tmp')
        else:
            output_base = self.config.get('run', {}).get('output_base', '/tmp')
            tmp_base = os.path.join(output_base, 'tmp')
        os.makedirs(tmp_base, exist_ok=True)

        if round_num is not None:
            dir_name = f"training_{name}_round_{round_num}"
        else:
            dir_name = f"training_{name}"
        
        output_dir = os.path.join(tmp_base, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
