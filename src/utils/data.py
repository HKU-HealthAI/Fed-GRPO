"""
Data utilities
"""

from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """
    Dataset wrapper for prompts, compatible with TRL's GRPOTrainer.
    
    Can be used for both federated and centralized training.
    """

    def __init__(self, data: List[Dict[str, Any]], tokenizer: Optional[Any] = None):
        """
        Args:
            data: List of samples, each containing 'prompt' key and other fields
            tokenizer: Optional tokenizer (for future use)
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return all columns - TRL will pass them to reward function
        return self.data[idx]
