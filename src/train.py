#!/usr/bin/env python3
"""
Training Script: Supports Fed-GRPO and baseline methods
"""

import os
import sys
import argparse
import torch
import torch.distributed
import gc
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Tuple
import copy
import logging

from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import FedGRPOServer, GRPOClient
from core.compression import compress_delta, decompress_to_state_dict
from dataloaders import load_federated_dataset
from rewards import create_reward_model
from utils import (
    set_seed, setup_logging, TrainingMetricsLogger, load_config,
    get_synchronized_timestamp, generate_experiment_name,
    setup_output_directory, print_experiment_info,
    save_model_checkpoint, is_main_process
)

logger = logging.getLogger(__name__)

# ===========================
# Training Method Constants
# ===========================

class TrainingMethod:
    """Training method constants to avoid magic strings."""
    FED_GRPO = 'fed_grpo'
    FEDAVG = 'fedavg'
    LOCAL_ONLY = 'local_only'
    CENTRALIZED = 'centralized'

    ALL = [FED_GRPO, FEDAVG, LOCAL_ONLY, CENTRALIZED]
    FEDERATED = [FED_GRPO, FEDAVG, LOCAL_ONLY]  # Methods that use federated data split


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fed-GRPO Unified Training Script")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--method", type=str, default=None,
                        choices=TrainingMethod.ALL,
                        help=f"Training method (override config): {', '.join(TrainingMethod.ALL)}")
    parser.add_argument("--aggregation", type=str, default=None,
                        choices=('sample_weighted', 'uniform'),
                        help="Aggregation method for fedavg (override config): sample_weighted | uniform")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device (override config): cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (override config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (override config)")
    return parser.parse_args()


def load_model_and_tokenizer(
    config: Dict[str, Any], 
    device: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """Load model and tokenizer
    
    Args:
        config: Configuration dictionary
        device: Target device (cuda or cpu)
        
    Returns:
        Tuple of (model, tokenizer, actual_device)
    """
    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)
    
    # Check if offline mode is enabled (for compute nodes without internet)
    local_files_only = os.environ.get('HF_HUB_OFFLINE', '0') == '1' or \
                       os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side='left',
        local_files_only=local_files_only
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    training_config = config.get('training', {})
    use_bf16 = training_config.get('bf16', True)
    if device == 'cuda':
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only
    )
    
    # Use Accelerator's device if available, otherwise use specified device
    # This ensures model goes to correct GPU in multi-GPU setup
    try:
        from accelerate import PartialState
        accelerator_device = PartialState().device
        model = model.to(accelerator_device)
        device = str(accelerator_device)  # Update device for downstream use
    except (ImportError, RuntimeError) as e:
        if is_main_process():
            logger.debug(f"PartialState not available, using specified device: {e}")
        model = model.to(device)
    
    # Log model info, only on the main process in distributed training
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Model cache dir: {config['model'].get('cache_dir', 'default')}")
        logger.info(f"Model device: {device}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer, device


def load_federated_dataset_and_reward(
    config: Dict[str, Any], 
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dict[str, List], Any]:
    """Load federated dataset and reward model
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer (for future use)
        
    Returns:
        Tuple of (federated_data, reward_model)
    """
    dataset_config = config['dataset']
    
    # Load federated dataset
    federated_data = load_federated_dataset(
        dataset_name=dataset_config['name'],
        num_clients=dataset_config['num_clients'],
        alpha=dataset_config.get('alpha', 0.5),
        split_method=dataset_config.get('split_method', 'dirichlet'),
        seed=config.get('seed', 42),
        cache_dir=dataset_config.get('cache_dir', None),
        hard_only=dataset_config.get('hard_only', False)
    )
    
    # Create reward model
    training_config = config.get('training', {})
    use_format_reward = training_config.get('use_format_reward', True)
    reward_model = create_reward_model(
        dataset_name=dataset_config['name'],
        use_format_reward=use_format_reward,
    )
    
    return federated_data, reward_model


def load_centralized_dataset_and_reward(
    config: Dict[str, Any]
) -> Tuple[List[Dict], Any]:
    """Load complete dataset directly (no split), for centralized training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_data, reward_model)
    """
    from dataloaders import load_centralized_dataset
    
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    cache_dir = dataset_config.get('cache_dir', None)
    
    # Use centralized loader from datasets module
    train_data = load_centralized_dataset(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        hard_only=dataset_config.get('hard_only', False)
    )
    
    # Create reward model
    training_config = config.get('training', {})
    use_format_reward = training_config.get('use_format_reward', True)
    reward_model = create_reward_model(
        dataset_name=dataset_name,
        use_format_reward=use_format_reward,
    )
    
    return train_data, reward_model


# ===========================
# Unified Federated Training
# ===========================

def _run_federated_training(
    config: Dict[str, Any],
    device: str,
    output_dir: str,
    method: str = 'fed_grpo'
) -> None:
    """Unified federated training function for Fed-GRPO and FedAvg.

    Args:
        config: Configuration dictionary
        device: Training device
        output_dir: Output directory
        method: Training method ('fed_grpo' | 'fedavg')
    """
    method_display_map = {
        'fed_grpo': 'Fed-GRPO',
        'fedavg': 'FedAvg (Federated Averaging)',
    }
    method_display = method_display_map.get(method, method)
    log_dir = config.get('_log_dir', os.path.join(output_dir, 'logs'))
    logger = setup_logging(log_dir)
    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {method_display} Training")
        logger.info(f"{'='*60}\n")
        logger.info(f"Config: {config}")

    # Set random seed
    set_seed(config.get('seed', 42))

    # Load model and data
    model, tokenizer, device = load_model_and_tokenizer(config, device)
    federated_data, reward_model = load_federated_dataset_and_reward(config, tokenizer)

    # Get config sections
    federated_config = config.get('federated', {})
    training_config = config.get('training', {})
    beta_kl = training_config.get('beta_kl', 0.001)
    num_rounds = federated_config.get('num_rounds', 50)
    ref_model_update_interval = federated_config.get('ref_model_update_interval', 10)
    ref_model_mixup_alpha = federated_config.get('ref_model_mixup_alpha', 1.0)

    # Aggregation: from YAML federated.aggregation (Fed-GRPO => signal_weighted; others => sample_weighted | uniform)
    if method == 'fed_grpo':
        aggregation_method = 'signal_weighted'
    else:
        aggregation_method = federated_config.get('aggregation', 'sample_weighted')
        if aggregation_method not in ('sample_weighted', 'uniform'):
            aggregation_method = 'sample_weighted'

    # Get checkpoint save frequency for FL (round-based)
    save_rounds_freq = federated_config.get('save_rounds_freq', 0)

    # Only create ref_model when beta_kl > 0 (for KL divergence computation)
    use_ref_model = beta_kl > 0
    server = FedGRPOServer(
        global_model=model,
        aggregation_method=aggregation_method,
        ref_model_update_interval=ref_model_update_interval,
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        use_ref_model=use_ref_model,
        use_effective_samples=federated_config.get('use_effective_samples', True),
        use_update_normalization=federated_config.get('use_update_normalization', True),
        calibration_alpha=federated_config.get('calibration_alpha', 0.5)
    )

    # Store client data
    client_data_list = federated_data['train']
    num_clients = len(client_data_list)

    # Safety check
    if num_clients == 0:
        if is_main_process():
            logger.error("No clients available for training!")
        raise ValueError("num_clients must be > 0")

    # Initialize metrics logger
    metrics_logger = TrainingMetricsLogger(
        log_dir=log_dir,
        method=method,
        num_rounds=num_rounds,
        num_clients=num_clients
    )

    if is_main_process():
        logger.info(f"Training with {num_clients} clients for {num_rounds} rounds")

    # Initialize reference model for KL divergence computation (only when beta_kl > 0)
    ref_weights = server.get_ref_weights() if use_ref_model else None

    # Global reward calibration settings
    use_global_reward_calibration = federated_config.get('use_global_reward_calibration', False)

    # Communication compression settings
    compression_config = federated_config.get('compression', {})
    use_compression = compression_config.get('use_compression', False)
    prev_round_reward_stds = None  # For adaptive mode: use previous round's reward_stds

    if is_main_process() and use_compression:
        logger.info(f"Communication compression enabled: budget={compression_config.get('budget', 1.0)}, "
                     f"adaptive={compression_config.get('adaptive', False)}, "
                     f"adaptive_alpha={compression_config.get('adaptive_alpha', 0.5)}")

    # Federated training loop
    for round_idx in range(num_rounds):
        if is_main_process():
            logger.info(f"\n{'='*60}")
            logger.info(f"Federation Round {round_idx + 1}/{num_rounds}")
            logger.info(f"{'='*60}")

        metrics_logger.start_round(round_idx)

        # Client local training
        client_updates = []
        client_metrics = []

        for client_id in range(num_clients):
            # Create fresh model copy for each client to avoid stale DDP/NCCL state
            client_model = copy.deepcopy(server.global_model)

            client = GRPOClient(
                client_id=client_id,
                model=client_model,
                train_data=client_data_list[client_id],
                tokenizer=tokenizer,
                reward_model=reward_model,
                config=config,
                device=device,
                federation_round=round_idx + 1  # 1-indexed for readability
            )

            # Load reference model weights for KL divergence computation (only when beta_kl > 0)
            if ref_weights is not None:
                client.load_ref_weights(ref_weights)

            # Pass global reward stats for per-prompt calibration (not available in round 1)
            if use_global_reward_calibration:
                g_mean, g_std = server.get_global_reward_stats()
                if g_mean is not None:
                    client.set_global_reward_stats(g_mean, g_std)

            # Train and collect updates
            update, metrics = client.local_train()

            # Apply communication compression if enabled (Fed-GRPO native compressor)
            if use_compression:
                global_state_bf16 = {k: v.cpu().to(torch.bfloat16)
                                     for k, v in server.global_model.state_dict().items()}
                local_state_bf16 = {k: v.to(torch.bfloat16) for k, v in update.items()}

                compressed = compress_delta(
                    global_state_bf16, local_state_bf16,
                    reward_std=metrics.get('reward_std', 0.0),
                    compression_config=compression_config,
                    prev_reward_stds=prev_round_reward_stds,
                )
                update = decompress_to_state_dict(compressed, global_state_bf16)

                if is_main_process():
                    stats = compressed['stats']
                    logger.info(
                        f"  Client {client_id} compression: "
                        f"bf16_absorbed={stats['bf16_absorbed_frac']:.4f}, "
                        f"budget={stats['actual_budget']:.2f}, "
                        f"modules={stats['num_selected_modules']}/{stats['num_selected_modules']+stats['num_skipped_modules']}, "
                        f"||Δ||²_kept={stats['delta_preserved_frac']:.4f}, "
                        f"param_compress={stats['param_compression_ratio']:.0f}x"
                    )

                del global_state_bf16, local_state_bf16, compressed

            client_updates.append(update)
            client_metrics.append(metrics)

            # Log client metrics using unified logger
            metrics_logger.log_client(client_id, metrics)

            # Explicitly delete client and model to free memory and DDP state
            del client
            del client_model

            # Force synchronization and cleanup between clients
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(0.5)

        # Server aggregates updates
        agg_stats = server.aggregate(client_updates, client_metrics)

        # Update prev_round_reward_stds for adaptive compression in next round
        if use_compression and compression_config.get('adaptive', False):
            prev_round_reward_stds = [m.get('reward_std', 0.0) for m in client_metrics]

        # End round and log summary
        metrics_logger.end_round(agg_stats)

        # Update reference weights for next round (only when beta_kl > 0)
        if use_ref_model:
            ref_weights = server.get_ref_weights()

        # Save checkpoint every N rounds (1-indexed round number)
        round_num = round_idx + 1
        if save_rounds_freq > 0 and round_num % save_rounds_freq == 0:
            round_checkpoint_dir = os.path.join(output_dir, 'checkpoints', f'checkpoint_round_{round_num}')
            save_model_checkpoint(server.global_model, tokenizer, round_checkpoint_dir, logger)

    # Save training summary
    metrics_logger.save_summary()

    torch.cuda.empty_cache()

    # Save final model
    checkpoint_dir = os.path.join(output_dir, 'final_checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    server.global_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    if is_main_process():
        logger.info(f"{method_display} training completed. Checkpoint saved to {checkpoint_dir}")


def run_fed_grpo(config: Dict[str, Any], device: str, output_dir: str) -> None:
    """Fed-GRPO training main function"""
    _run_federated_training(config, device, output_dir, method='fed_grpo')


def run_fedavg(config: Dict[str, Any], device: str, output_dir: str) -> None:
    """FedAvg training: standard federated averaging"""
    _run_federated_training(config, device, output_dir, method='fedavg')


def run_local_only(config: Dict[str, Any], device: str, output_dir: str) -> None:
    """Local-Only training: each client trains independently with max_steps"""
    log_dir = config.get('_log_dir', os.path.join(output_dir, 'logs'))
    logger = setup_logging(log_dir)
    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Local-Only Training (each client independent)")
        logger.info(f"{'='*60}\n")

    set_seed(config.get('seed', 42))

    # Load model and data
    base_model, tokenizer, device = load_model_and_tokenizer(config, device)
    federated_data, reward_model = load_federated_dataset_and_reward(config, tokenizer)

    training_config = config.get('training', {})
    max_steps = training_config.get('max_steps', 500)

    # Each client trains independently
    num_clients = len(federated_data['train'])

    # Safety check
    if num_clients == 0:
        if is_main_process():
            logger.error("No clients available for training!")
        raise ValueError("num_clients must be > 0")

    # Initialize metrics logger
    metrics_logger = TrainingMetricsLogger(
        log_dir=log_dir,
        method='local_only',
        num_clients=num_clients,
        max_steps=max_steps
    )

    # Get checkpoint save frequency
    save_steps_freq = training_config.get('save_steps_freq', 0)

    if is_main_process():
        logger.info(f"Training {num_clients} clients independently, {max_steps} steps each")

    for client_id, client_data in enumerate(federated_data['train']):
        if is_main_process():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Client {client_id + 1}/{num_clients}, data size: {len(client_data)} samples")
            logger.info(f"{'='*60}")

        # Create independent model copy for each client
        client_model = copy.deepcopy(base_model)

        client = GRPOClient(
            client_id=client_id,
            model=client_model,
            train_data=client_data,
            tokenizer=tokenizer,
            reward_model=reward_model,
            config=config,
            device=device
        )

        # Train with max_steps (standard GRPO training)
        if is_main_process():
            logger.info(f"Client {client_id} training for {max_steps} steps...")
        
        # Set checkpoint base dir for this client
        checkpoint_base_dir = os.path.join(output_dir, f"client_{client_id}", "checkpoints") if save_steps_freq > 0 else None
        
        metrics = client.train(
            client_name=f"Client {client_id}",
            metrics_logger=metrics_logger,
            checkpoint_base_dir=checkpoint_base_dir
        )

        # Log training completion with unified logger
        metrics_logger.log_training_complete(metrics, client_name=f"Client {client_id}")

        # Save client checkpoint using helper function
        client_checkpoint_dir = os.path.join(output_dir, f"client_{client_id}")
        save_model_checkpoint(client.model, tokenizer, client_checkpoint_dir, logger)

        # Cleanup client resources
        del client
        del client_model

        # Force synchronization and cleanup
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Brief delay to allow cleanup to complete
        time.sleep(0.5)

        if is_main_process():
            logger.info(f"Client {client_id} training and checkpoint saving completed")

    # Save training summary
    metrics_logger.save_summary()

    # Free base model after all clients are done
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_main_process():
        logger.info(f"Local-only training completed. All client checkpoints saved to {output_dir}")
        logger.info(f"Saved checkpoints for {num_clients} clients")


def run_centralized(config: Dict[str, Any], device: str, output_dir: str) -> None:
    """Centralized training: standard GRPO training on all data"""
    log_dir = config.get('_log_dir', os.path.join(output_dir, 'logs'))
    logger = setup_logging(log_dir)
    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Centralized Training")
        logger.info(f"{'='*60}\n")

    set_seed(config.get('seed', 42))

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(config, device)

    # Load complete dataset directly (no splitting)
    train_data, reward_model = load_centralized_dataset_and_reward(config)

    if is_main_process():
        logger.info(f"Centralized training data size: {len(train_data)} samples")

    training_config = config.get('training', {})
    max_steps = training_config.get('max_steps', 500)

    # Initialize metrics logger
    metrics_logger = TrainingMetricsLogger(
        log_dir=log_dir,
        method='centralized',
        max_steps=max_steps
    )

    # Create trainer for centralized training
    central_trainer = GRPOClient(
        client_id=0,
        model=model,
        train_data=train_data,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=config,
        device=device
    )

    # Train with max_steps
    if is_main_process():
        logger.info(f"\nTraining for {max_steps} steps...")

    # Get checkpoint save frequency
    save_steps_freq = training_config.get('save_steps_freq', 0)
    checkpoint_base_dir = os.path.join(output_dir, "checkpoints") if save_steps_freq > 0 else None

    metrics = central_trainer.train(
        client_name="Centralized Server",
        metrics_logger=metrics_logger,
        checkpoint_base_dir=checkpoint_base_dir
    )

    # Log training completion with unified logger
    metrics_logger.log_training_complete(metrics, client_name="Centralized Server")

    # Save training summary
    metrics_logger.save_summary()

    # Save checkpoint using helper function
    checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    save_model_checkpoint(central_trainer.model, tokenizer, checkpoint_dir, logger)
    
    # Cleanup
    del central_trainer
    del model
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if is_main_process():
        logger.info(f"Centralized training completed. Checkpoint saved to {checkpoint_dir}")


# ===========================
# Main Function
# ===========================

def main() -> None:
    """Main entry point for training."""
    args = parse_args()

    # Pre-initialize NCCL process group with extended timeout (3600s).
    # GRPO generation causes large rank desync on PCIe-only clusters,
    # exceeding the default 600s NCCL operation timeout.
    # This must happen before any Accelerator / PartialState creation;
    # later Accelerator() calls see is_initialized()=True and reuse this group.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=3600),
        )

    # Load config
    config = load_config(args.config)
    run_config = config.get('run', {})
    
    # Determine parameters (CLI takes priority over config)
    method = args.method if args.method is not None else run_config.get('method', TrainingMethod.FED_GRPO)
    device = args.device if args.device is not None else run_config.get('device', 'cuda')
    seed = args.seed if args.seed is not None else run_config.get('seed', 42)
    config['seed'] = seed

    # Aggregation: CLI overrides YAML (only used for fedavg; fed_grpo always uses signal_weighted)
    aggregation = args.aggregation if args.aggregation is not None else config.get('federated', {}).get('aggregation', 'sample_weighted')
    if method == 'fedavg' and aggregation not in ('sample_weighted', 'uniform'):
        aggregation = 'sample_weighted'
    config.setdefault('federated', {})['aggregation'] = aggregation
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        exp_name = os.path.basename(output_dir.rstrip('/'))
    else:
        timestamp = get_synchronized_timestamp()
        # For fedavg, include aggregation (sample_weighted | uniform) in output_dir name
        fed_agg = None
        if method == 'fedavg':
            fed_agg = config.get('federated', {}).get('aggregation', 'sample_weighted')
            if fed_agg not in ('sample_weighted', 'uniform'):
                fed_agg = 'sample_weighted'
        exp_name = generate_experiment_name(
            config, method, timestamp, TrainingMethod.FEDERATED, aggregation=fed_agg
        )
        output_base = run_config.get('output_base', 'outputs')
        output_dir = os.path.join(output_base, exp_name)
    
    config['_log_dir'] = output_dir
    setup_output_directory(output_dir)
    setup_logging(output_dir)

    # Log experiment info (resolved run params + full config)
    print_experiment_info(exp_name, output_dir, method, device, seed, config)
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        if is_main_process():
            logger.warning("CUDA not available, switching to CPU")
        device = 'cpu'
    
    # Run training
    training_methods = {
        TrainingMethod.FED_GRPO: run_fed_grpo,
        TrainingMethod.LOCAL_ONLY: run_local_only,
        TrainingMethod.FEDAVG: run_fedavg,
        TrainingMethod.CENTRALIZED: run_centralized,
    }
    
    if method not in training_methods:
        raise ValueError(f"Unknown method: {method}. Available: {TrainingMethod.ALL}")
    
    training_methods[method](config, device, output_dir)
    
    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed! Results saved to: {output_dir}")
        logger.info(f"{'='*60}\n")

    # Clean up distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
