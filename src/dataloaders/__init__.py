"""
Dataset Loaders

Supports MATH dataset for training.
Supports GSM8K, MATH-500, AIME 2024, AIME 2025, AMC for evaluation.
"""

from typing import Dict, List, Type, Tuple
import logging
import numpy as np

from .math_loader import MATHDataset, EVAL_DATASETS, load_eval_dataset, load_all_eval_datasets
from .base_loader import BaseDataset

logger = logging.getLogger(__name__)

# ===========================
# Dataset Registry (Training)
# ===========================

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    'math': MATHDataset,
}


def get_dataset_class(dataset_name: str) -> Type[BaseDataset]:
    """
    Get dataset class by name.
    
    Args:
        dataset_name: Dataset name (case-insensitive)
        
    Returns:
        Dataset class
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    name_lower = dataset_name.lower()
    if name_lower not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name_lower]


def load_federated_dataset(
    dataset_name: str,
    num_clients: int,
    alpha: float = 0.5,
    split_method: str = 'dirichlet',
    seed: int = 42,
    cache_dir: str = None,
    hard_only: bool = False
) -> Dict[str, List]:
    """
    Load federated dataset.
    
    Args:
        dataset_name: Dataset name (math)
        num_clients: Number of clients
        alpha: Dirichlet parameter (only for dirichlet method)
        split_method: Data splitting method
        seed: Random seed
        cache_dir: Dataset cache directory
        hard_only: If True, only keep Level 3-5 problems (default: False)
    
    Returns:
        {'train': [client1_data, client2_data, ...]}
    """
    np.random.seed(seed)
    
    # Get dataset class from registry
    DatasetClass = get_dataset_class(dataset_name)
    
    # Create dataset (__init__ already calls load_data())
    dataset = DatasetClass(num_clients=num_clients, alpha=alpha, cache_dir=cache_dir, hard_only=hard_only)
    
    # Split data
    dataset.split_data(method=split_method)
    
    # After split, num_clients may have been adjusted (e.g., difficulty split)
    actual_num_clients = dataset.num_clients
    
    # Prepare return result
    federated_data = {
        'train': [dataset.get_client_data(i) for i in range(actual_num_clients)],
    }
    
    return federated_data


def load_centralized_dataset(
    dataset_name: str,
    cache_dir: str = None,
    hard_only: bool = False
) -> List[Dict]:
    """
    Load complete dataset directly (no split), for centralized training.
    
    Args:
        dataset_name: Dataset name (case-insensitive)
        cache_dir: Dataset cache directory
        hard_only: If True, only keep Level 3-5 problems (default: False)
        
    Returns:
        train_data: List of training samples
    """
    # Get dataset class from registry
    DatasetClass = get_dataset_class(dataset_name)
    
    # Create dataset with minimal config (single client)
    dataset = DatasetClass(num_clients=1, alpha=1.0, cache_dir=cache_dir, hard_only=hard_only)
    
    # Get all training data
    train_data = dataset.data
    
    from utils.experiment import is_main_process
    if is_main_process():
        logger.info(f"Centralized training data: {len(train_data)} samples")
    
    return train_data


__all__ = [
    # Dataset classes
    "MATHDataset",
    "BaseDataset",
    # Registry and utilities
    "DATASET_REGISTRY",
    "get_dataset_class",
    "load_federated_dataset",
    "load_centralized_dataset",
    # Evaluation datasets
    "EVAL_DATASETS",
    "load_eval_dataset",
    "load_all_eval_datasets",
]
