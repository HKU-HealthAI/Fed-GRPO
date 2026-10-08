"""
Experiment setup utilities
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.distributed

logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    """Check if current process is the main process.
    
    Returns:
        True if main process, False otherwise
    """
    try:
        from accelerate import PartialState
        state = PartialState()
        return state.is_main_process
    except (ImportError, RuntimeError):
        # If accelerate not available, assume single process (main)
        return True


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_synchronized_timestamp() -> str:
    """Get timestamp synchronized across all processes in distributed training.
    
    In multi-GPU/multi-process training, ensures all processes use the same timestamp
    by broadcasting from the main process.
    
    Returns:
        Timestamp string in format "YYYYMMDD_HHMMSS"
    """
    try:
        from accelerate import PartialState
        state = PartialState()
        
        # Only main process generates timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if state.is_main_process else None
        
        # Broadcast timestamp in distributed setting
        if torch.distributed.is_initialized():
            if state.is_main_process:
                timestamp_bytes = timestamp.encode('utf-8')
                timestamp_len = len(timestamp_bytes)
                timestamp_tensor = torch.zeros(64, dtype=torch.uint8, device=state.device)
                timestamp_tensor[:timestamp_len] = torch.tensor(
                    list(timestamp_bytes), dtype=torch.uint8, device=state.device
                )
                len_tensor = torch.tensor([timestamp_len], dtype=torch.long, device=state.device)
            else:
                timestamp_tensor = torch.zeros(64, dtype=torch.uint8, device=state.device)
                len_tensor = torch.tensor([0], dtype=torch.long, device=state.device)
            
            torch.distributed.broadcast(len_tensor, src=0)
            torch.distributed.broadcast(timestamp_tensor, src=0)
            
            timestamp_len = len_tensor.item()
            timestamp_bytes = bytes(timestamp_tensor[:timestamp_len].cpu().numpy())
            timestamp = timestamp_bytes.decode('utf-8')
        elif timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
    except (ImportError, RuntimeError):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return timestamp


def extract_model_name(raw_model_name: str) -> str:
    """Extract clean model name from path or HuggingFace identifier.
    
    Examples:
        "Qwen/Qwen2.5-3B" -> "Qwen2.5-3B"
        "/path/to/model--Qwen--Qwen2.5-3B" -> "Qwen2.5-3B"
        "Qwen2.5-3B" -> "Qwen2.5-3B"
    
    Args:
        raw_model_name: Model path or identifier
        
    Returns:
        Clean model name
    """
    if '/' in raw_model_name:
        model_name = raw_model_name.split('/')[-1]
        if '--' in model_name:
            model_name = model_name.split('--')[-1]
    else:
        model_name = raw_model_name
    return model_name


def generate_experiment_name(
    config: Dict[str, Any],
    method: str,
    timestamp: str,
    federated_methods: Optional[List[str]] = None,
    aggregation: Optional[str] = None
) -> str:
    """Generate descriptive experiment name.
    
    Format:
      - Federated: {model}_{method}[_{aggregation}]_{beta_kl}_{lr}_{noniid_info}_{timestamp}
        For fedavg, aggregation (sample_weighted | uniform) is appended right after method.
      - Centralized: {model}_{method}_{beta_kl}_{lr}_{timestamp}
    
    Args:
        config: Configuration dictionary
        method: Training method
        timestamp: Timestamp string
        federated_methods: List of method names considered "federated" (for non-IID info)
        aggregation: For fedavg only: 'sample_weighted' or 'uniform', included in name when set.
        
    Returns:
        Experiment name string
    """
    if federated_methods is None:
        federated_methods = ['fed_grpo', 'fedavg', 'local_only']
    
    dataset_config = config.get('dataset', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    model_name = extract_model_name(model_config.get('name', 'unknown'))
    split_method = dataset_config.get('split_method', 'random')
    num_clients = dataset_config.get('num_clients', 1)
    alpha = dataset_config.get('alpha', 1.0)
    beta_kl = training_config.get('beta_kl', 0.01)
    learning_rate = training_config.get('learning_rate', None)
    
    beta_kl_str = f"kl{beta_kl}"
    lr_str = f"lr{learning_rate}" if learning_rate is not None else "lrNA"
    method_part = f"{method}_{aggregation}" if (method == 'fedavg' and aggregation) else method

    if method in federated_methods:
        # Include non-IID information for federated methods; for fedavg aggregation is after method
        if split_method == 'dirichlet':
            noniid_info = f"dir{alpha}_c{num_clients}"
        else:
            noniid_info = f"{split_method}_c{num_clients}"
        return f"{model_name}_{method_part}_{beta_kl_str}_{lr_str}_{noniid_info}_{timestamp}"
    else:
        # Centralized doesn't need non-IID info
        return f"{model_name}_{method}_{beta_kl_str}_{lr_str}_{timestamp}"


def setup_output_directory(output_dir: str) -> None:
    """Create output directory, handling distributed training race conditions.
    
    In distributed training, only the main process creates the directory,
    then all processes synchronize.
    
    Args:
        output_dir: Path to output directory
    """
    try:
        from accelerate import PartialState
        state = PartialState()
        if state.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except (ImportError, RuntimeError):
        os.makedirs(output_dir, exist_ok=True)


def print_experiment_info(
    exp_name: str,
    output_dir: str,
    method: str,
    device: str,
    seed: int,
    config: Dict[str, Any]
) -> None:
    """Log experiment configuration: resolved run params and full config hyperparameters.

    Only runs on the main process in distributed training.

    Args:
        exp_name: Experiment name
        output_dir: Output directory path
        method: Training method
        device: Training device
        seed: Random seed
        config: Full configuration dictionary (all hyperparameters)
    """
    if not is_main_process():
        return
    logger.info(
        "Experiment: exp_name=%s, output_dir=%s, method=%s, device=%s, seed=%s",
        exp_name, output_dir, method, device, seed
    )
    try:
        config_str = json.dumps(config, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        config_str = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Full config:\n%s", config_str)
