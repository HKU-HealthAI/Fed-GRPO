"""
Fed-GRPO: Federated Group Relative Policy Optimization
A framework for federated fine-tuning of Large Language Models using GRPO.
"""

__version__ = "0.2.0"

# Core components
from .core import FedGRPOServer, GRPOClient

# Dataset utilities
from .dataloaders import (
    load_federated_dataset,
    load_centralized_dataset,
    get_dataset_class,
    DATASET_REGISTRY,
)

# Reward models
from .rewards import create_reward_model, RewardModel

# Utilities
from .utils import (
    set_seed,
    setup_logging,
    load_config,
    PromptDataset,
    TrainingMetricsLogger,
    # Answer extraction utilities
    extract_answer,
    normalize_answer,
)

# Training method constants (from train.py, re-exported for convenience)
# Note: Import at runtime to avoid circular imports
def _get_training_method():
    """Lazy import of TrainingMethod to avoid circular imports."""
    from .train import TrainingMethod
    return TrainingMethod


__all__ = [
    # Core
    "FedGRPOServer",
    "GRPOClient",
    # Datasets
    "load_federated_dataset",
    "load_centralized_dataset",
    "get_dataset_class",
    "DATASET_REGISTRY",
    # Rewards
    "create_reward_model",
    "RewardModel",
    # Utilities
    "set_seed",
    "setup_logging",
    "load_config",
    "PromptDataset",
    "TrainingMetricsLogger",
    "extract_answer",
    "normalize_answer",
]
