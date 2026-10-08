"""
Dataset base class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from utils.experiment import is_main_process

logger = logging.getLogger(__name__)


# ===========================
# R1-Style Prompt Templates
# ===========================

class PromptTemplates:
    """
    Unified prompt templates for R1-style math training.
    
    Uses DeepSeek-R1 style format with <think> and <answer> tags.
    """
    
    # Base template parts
    _PREFIX = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    )
    
    @classmethod
    def math_symbolic(cls, problem: str) -> str:
        """Unified template for all math problems."""
        return (
            cls._PREFIX +
            "<answer> \\boxed{answer} </answer>.\n"
            f"User: {problem}\n"
            "Assistant: <think>"
        )


class BaseDataset(ABC):
    """
    Dataset base class.
    
    Provides common functionality for all dataset loaders:
    - Prompt templates (R1-style)
    - Dirichlet non-IID data splitting
    - Random IID data splitting
    - Client data management
    """
    
    # Class-level prompt templates
    templates = PromptTemplates
    
    def __init__(self, num_clients: int, alpha: float = 0.1, cache_dir: str = None):
        """
        Args:
            num_clients: Number of clients
            alpha: Dirichlet distribution parameter, smaller means more Non-IID
            cache_dir: Dataset cache directory
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.cache_dir = cache_dir
        self.data: List[Dict[str, Any]] = []  # Training data
        self.test_data: List[Dict[str, Any]] = []  # Test data
        self.client_data: List[List[Dict[str, Any]]] = [[] for _ in range(num_clients)]
    
    @abstractmethod
    def load_data(self) -> None:
        """Load data. Must be implemented by subclasses."""
        pass
    
    def split_data(self, method: str = "dirichlet") -> None:
        """
        Split data for each client.
        
        Default implementation supports:
        - "dirichlet": Non-IID split using Dirichlet distribution (requires labels)
        - "random": IID random uniform split
        
        Subclasses can override to add dataset-specific split methods.
        
        Args:
            method: Split method name
        """
        if method == "random":
            client_indices = self.random_split()
        else:
            # For dirichlet and other methods, subclasses should override
            raise NotImplementedError(
                f"Split method '{method}' not implemented in base class. "
                f"Subclass should override split_data() to handle this method."
            )
        
        # Allocate data to clients
        self._allocate_data(client_indices)
    
    def random_split(self) -> List[List[int]]:
        """
        Random IID uniform split.
        
        Returns:
            List of data indices for each client
        """
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, self.num_clients)
        return [idx.tolist() for idx in client_indices]
    
    def _do_dirichlet_split(self, labels: List[int], unique_labels: np.ndarray) -> List[List[int]]:
        """Single attempt of Dirichlet split."""
        client_indices = [[] for _ in range(self.num_clients)]

        for label in unique_labels:
            idx_label = np.where(np.array(labels) == label)[0]
            np.random.shuffle(idx_label)

            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(idx_label)).astype(int)[:-1]
            idx_split = np.split(idx_label, proportions)

            for i, idx in enumerate(idx_split):
                client_indices[i].extend(idx.tolist())

        return client_indices

    def dirichlet_split(self, labels: List[int], max_retries: int = 100) -> List[List[int]]:
        """
        Use Dirichlet distribution for Non-IID data split.

        Args:
            labels: List of data labels (determines non-IID distribution)
            max_retries: Maximum retry attempts if empty clients exist

        Returns:
            List of data indices for each client
        """
        n_samples = len(labels)
        unique_labels = np.unique(labels)

        # Try Dirichlet split with retries
        for attempt in range(max_retries):
            client_indices = self._do_dirichlet_split(labels, unique_labels)
            empty_clients = [i for i in range(self.num_clients) if len(client_indices[i]) == 0]

            if not empty_clients:
                if attempt > 0 and is_main_process():
                    logger.info(f"Dirichlet split succeeded after {attempt + 1} attempts")
                break
        else:
            # Exceeded max retries, fallback to borrowing
            if is_main_process():
                logger.warning(
                    f"Dirichlet split still has empty clients after {max_retries} attempts. "
                    f"Falling back to borrowing samples."
                )
            empty_clients = [i for i in range(self.num_clients) if len(client_indices[i]) == 0]
            min_samples = max(1, n_samples // (self.num_clients * 10))

            for empty_client in empty_clients:
                donor = max(range(self.num_clients), key=lambda i: len(client_indices[i]))
                if len(client_indices[donor]) > min_samples:
                    n_transfer = min(min_samples, len(client_indices[donor]) // 2)
                    client_indices[empty_client] = client_indices[donor][:n_transfer]
                    client_indices[donor] = client_indices[donor][n_transfer:]
                    if is_main_process():
                        logger.info(f"Transferred {n_transfer} samples: client {donor} -> {empty_client}")

        # Shuffle each client data
        for i in range(self.num_clients):
            np.random.shuffle(client_indices[i])

        return client_indices
    
    def _allocate_data(self, client_indices: List[List[int]]) -> None:
        """
        Allocate data to clients based on indices.
        
        Args:
            client_indices: List of data indices for each client
        """
        for client_id, indices in enumerate(client_indices):
            self.client_data[client_id] = [self.data[i] for i in indices]
            if is_main_process():
                logger.info(f"Client {client_id}: {len(self.client_data[client_id])} samples")
    
    def get_client_data(self, client_id: int) -> List[Dict[str, Any]]:
        """Get training data for specified client."""
        return self.client_data[client_id]
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get all test data."""
        return self.test_data.copy() if self.test_data else []
    
    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample by index."""
        return self.data[idx]
