"""
Reward Functions Module

All math-related datasets use the unified math reward function.

Supports two modes:
1. Accuracy-only reward (default): 1.0 for correct, 0.0 for incorrect
2. Accuracy + Format reward: Adds bonus for R1-style <think>...</think><answer>...</answer> format

Reference: DeepSeek-R1 uses format reward to encourage proper reasoning structure.
"""

from .math_reward import (
    math_reward_function,
    math_reward_with_format,
    format_reward,
    check_r1_format,
)


class RewardModel:
    """Reward Model Wrapper (DeepSeek-R1-Zero style)
    
    Uses Reward_rule = Reward_acc + Reward_format with equal weights.
    
    Args:
        reward_func: Base reward function
        reward_type: 'accuracy' or 'accuracy_with_format'
    """
    
    def __init__(self, reward_func, reward_type: str = 'accuracy'):
        self.reward_func = reward_func
        self.reward_type = reward_type
    
    def compute_rewards(self, responses, references):
        """Compute rewards for a batch of responses"""
        if not isinstance(responses, list):
            responses = [responses]
        if not isinstance(references, list):
            references = [references]
        
        rewards = []
        for response, reference in zip(responses, references):
            reward = self.reward_func(response, reference)
            rewards.append(reward)
        
        return rewards


def create_reward_model(
    dataset_name: str = 'math',
    reward_type: str = 'accuracy',
    use_format_reward: bool = True,
):
    """
    Create reward model for math datasets (DeepSeek-R1-Zero style).
    
    All supported datasets use the same math reward function:
    - math (training)
    - gsm8k, math500, aime2024, aime2025, amc (evaluation)
    
    Uses Reward_rule = Reward_acc + Reward_format with equal weights when use_format_reward=True.
    Answer extraction from <answer>...</answer> tags.
    
    Args:
        dataset_name: Dataset name (all use math reward)
        reward_type: Reward type (default 'accuracy')
        use_format_reward: Whether to add format reward (default True)
    
    Returns:
        RewardModel object
    """
    # All math datasets use the same reward function
    if use_format_reward:
        base_reward_func = math_reward_with_format
    else:
        base_reward_func = math_reward_function
    
    def reward_func(model_output: str, ground_truth: str) -> float:
        return base_reward_func(model_output, ground_truth)
    
    return RewardModel(reward_func, reward_type)


__all__ = [
    # MATH rewards (handles all math datasets)
    "math_reward_function", 
    "math_reward_with_format",
    # Format reward utilities
    "format_reward",
    "check_r1_format",
    # Reward model wrapper
    "RewardModel",
    "create_reward_model"
]
