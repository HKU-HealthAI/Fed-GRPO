"""
Test script for evaluating trained models using vLLM with multi-GPU acceleration
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from dataloaders import load_eval_dataset, EVAL_DATASETS
from rewards import create_reward_model
from utils import (
    setup_logging,
    extract_answer,
    get_answer_extractor,
    is_main_process
)

# os.environ["VLLM_USE_V1"] = "0"

logger = logging.getLogger(__name__)


# ===========================
# Evaluation Helper Functions
# ===========================

def _create_sampling_params(eval_mode: str, eval_config: Dict) -> Tuple[Any, int]:
    """Create vLLM sampling parameters based on evaluation mode.
    
    Args:
        eval_mode: Evaluation mode (greedy, pass@1, avg@k, maj@k)
        eval_config: Evaluation configuration
        
    Returns:
        Tuple of (SamplingParams, k) where k is number of samples per prompt
    """
    from vllm import SamplingParams
    
    num_samples_per_prompt = eval_config.get('num_samples_per_prompt', 32)
    temperature = eval_config.get('temperature', 0.6)
    top_p = eval_config.get('top_p', 0.95)
    max_new_tokens = eval_config.get('max_new_tokens', 512)
    
    if eval_mode == 'greedy':
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            n=1,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        k = 1
    else:
        k = num_samples_per_prompt
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=k,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
    
    return sampling_params, k


def _prepare_eval_data(
    eval_data: List[Dict],
    max_samples: Optional[int]
) -> Tuple[List[str], List[Any]]:
    """Prepare prompts and ground truths from evaluation data.
    
    Args:
        eval_data: Raw evaluation data
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Tuple of (prompts, ground_truths)
    """
    eval_samples = eval_data[:max_samples] if max_samples and len(eval_data) > max_samples else eval_data
    
    all_prompts = []
    all_ground_truths = []
    
    for s in eval_samples:
        prompt = s.get('prompt', s.get('question', ''))
        gt = s.get('answer', s.get('reference', ''))
        all_prompts.append(prompt)
        all_ground_truths.append(gt)
    
    return all_prompts, all_ground_truths


def _compute_greedy_metrics(
    responses: List[str],
    ground_truth: Any,
    reward_model: Any
) -> Tuple[int, int]:
    """Compute metrics for greedy evaluation mode.
    
    Returns:
        Tuple of (correct_count, total_samples)
    """
    response = responses[0]
    rewards = reward_model.compute_rewards([response], [ground_truth])
    correct = 1 if rewards[0] > 0.5 else 0
    return correct, 1


def _compute_avg_metrics(
    responses: List[str],
    ground_truth: Any,
    reward_model: Any,
    k: int
) -> Tuple[int, int, int]:
    """Compute metrics for pass@1/avg@k evaluation mode.
    
    Returns:
        Tuple of (total_correct_responses, total_responses, total_samples)
    """
    rewards = reward_model.compute_rewards(responses, [ground_truth] * k)
    sample_correct = sum(1 for r in rewards if r > 0.5)
    return sample_correct, k, 1


def _compute_majority_metrics(
    responses: List[str],
    ground_truth: Any,
    reward_model: Any,
    extract_func: Any = None,
) -> Tuple[int, int]:
    """Compute metrics for majority voting evaluation mode.
    
    Args:
        responses: List of model responses
        ground_truth: Ground truth answer
        reward_model: Reward model for verification
        extract_func: Dataset-specific answer extraction function
        
    Returns:
        Tuple of (correct_count, total_samples)
    """
    # Use provided extract_func
    if extract_func is None:
        if is_main_process():
            logger.warning("No dataset-specific answer extraction function provided, using generic extract_answer function.")
        return 0, 1
    
    answer_counts = Counter()
    for resp in responses:
        # Extract answer using dataset-specific function
        extracted = extract_func(resp)
        if extracted:
            answer_counts[extracted] += 1
    
    if not answer_counts:
        return 0, 1
    
    majority_answer, _ = answer_counts.most_common(1)[0]
    
    # Verify majority answer against ground truth
    # Use reward_model for accurate comparison (handles numerical tolerance, etc.)
    for resp in responses:
        if extract_func(resp) == majority_answer:
            rewards = reward_model.compute_rewards([resp], [ground_truth])
            if rewards[0] > 0.5:
                return 1, 1
            break
    
    return 0, 1


def _cleanup_vllm(llm: Any) -> None:
    """Clean up vLLM resources."""
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model_vllm(
    model_path: str,
    eval_data: List[Dict],
    reward_model: Any,
    device: str,
    eval_config: Optional[Dict] = None,
    num_gpus: int = 4,
    dataset_name: str = None,
) -> Dict[str, Any]:
    """
    Evaluate model using vLLM with multi-GPU acceleration.
    
    Args:
        model_path: Path to the trained model checkpoint
        eval_data: Evaluation dataset
        reward_model: Reward model for scoring
        device: Device for reward computation
        eval_config: Evaluation configuration
        num_gpus: Number of GPUs for vLLM tensor parallelism
        dataset_name: Dataset name for dataset-specific answer extraction
    
    Returns:
        Dictionary with evaluation metrics
    """
    from tqdm import tqdm
    logger = logging.getLogger(__name__)
    
    try:
        from vllm import LLM
    except ImportError:
        if is_main_process():
            logger.error("vLLM not available! Please install vllm.")
        sys.exit(1)
    
    # Default configuration
    eval_config = eval_config or {}
    eval_mode = eval_config.get('eval_mode', 'avg@32')
    max_samples = eval_config.get('max_samples', None)
    gpu_memory_utilization = eval_config.get("gpu_memory_utilization", 0.9)
    
    # Get dataset-specific answer extractor
    extract_func = get_answer_extractor(dataset_name) if dataset_name else None
    
    # Prepare data
    all_prompts, all_ground_truths = _prepare_eval_data(eval_data, max_samples)
    
    if len(all_prompts) == 0:
        if is_main_process():
            logger.warning("Evaluation dataset is empty")
        return {'accuracy': 0.0, 'correct': 0, 'total_samples': 0, 'eval_mode': eval_mode}
    
    # Initialize vLLM
    if is_main_process():
        logger.info(f"Loading model from {model_path} for vLLM evaluation...")
        logger.info(f"Using {num_gpus} GPUs with tensor parallelism")
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=num_gpus,
    )
    
    # Create sampling parameters
    sampling_params, k = _create_sampling_params(eval_mode, eval_config)
    
    # Generate responses
    if is_main_process():
        logger.info(f"Generating responses for {len(all_prompts)} prompts (mode={eval_mode}, k={k})...")
        logger.info(f"Total responses to generate: {len(all_prompts) * k}")
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)
    if is_main_process():
        logger.info("Generation completed! Processing results...")
    
    # Process results based on eval_mode
    correct_count = 0
    total_samples = 0
    total_correct_responses = 0
    total_responses = 0
    
    for i, output in enumerate(tqdm(outputs, desc="Processing results", unit="samples")):
        ground_truth = all_ground_truths[i]
        prompt = all_prompts[i]
        responses = [o.text for o in output.outputs]
        
        # Log detailed information every 50 samples
        if is_main_process() and i % 50 == 0:
            logger.info(f"Sample #{i} Output Information:")
            logger.info(f"--- Response #1 (of {len(responses)}) ---")
            if len(responses[0]) > 500:
                logger.info(f"{responses[0][:200]} ... [truncated] ... {responses[0][-200:]}")
            else:
                logger.info(responses[0])
            
            # Extract and log answer using dataset-specific extractor
            extracted_answer = extract_func(responses[0]) if extract_func else None
            logger.info(f"Extracted Answer: '{extracted_answer}'")
            logger.info(f"Ground Truth: '{ground_truth}'")
        
        if eval_mode == 'greedy':
            correct, samples = _compute_greedy_metrics(responses, ground_truth, reward_model)
            correct_count += correct
            total_samples += samples
            
        elif eval_mode == 'pass@1' or eval_mode.startswith('avg@'):
            correct_resp, resp_count, samples = _compute_avg_metrics(
                responses, ground_truth, reward_model, k
            )
            total_correct_responses += correct_resp
            total_responses += resp_count
            total_samples += samples
            
        elif eval_mode.startswith('maj@'):
            correct, samples = _compute_majority_metrics(responses, ground_truth, reward_model, extract_func)
            correct_count += correct
            total_samples += samples
            
        else:
            if is_main_process():
                logger.error(f"Unknown eval_mode '{eval_mode}'. Supported: greedy, pass@1, avg@k, maj@k")
            _cleanup_vllm(llm)
            sys.exit(1)
    
    # Cleanup
    _cleanup_vllm(llm)
    
    # Build result
    if eval_mode == 'greedy':
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        return {
            'eval_mode': 'greedy',
            'accuracy': accuracy,
            'correct': correct_count,
            'total_samples': total_samples
        }
    elif eval_mode == 'pass@1' or eval_mode.startswith('avg@'):
        avg_accuracy = total_correct_responses / total_responses if total_responses > 0 else 0.0
        return {
            'eval_mode': eval_mode,
            'accuracy': avg_accuracy,
            'total_correct_responses': total_correct_responses,
            'total_responses': total_responses,
            'total_samples': total_samples,
            'k': k
        }
    else:  # maj@k
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        return {
            'eval_mode': eval_mode,
            'accuracy': accuracy,
            'correct': correct_count,
            'total_samples': total_samples,
            'k': k
        }


def main():
    parser = argparse.ArgumentParser(description='Test trained models with vLLM acceleration')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Dataset to evaluate on (overrides config). Options: math500, gsm8k, aime2024, aime2025, amc')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs for vLLM tensor parallelism')
    parser.add_argument('--output', type=str, default=None, help='Output file for results (optional)')
    
    args = parser.parse_args()
    
    # Setup logging using unified helper
    if args.output:
        log_dir = Path(args.output).parent / 'eval_logs'
    else:
        checkpoint_path = Path(args.checkpoint)
        log_dir = checkpoint_path.parent / 'eval_logs' if checkpoint_path.exists() else Path('/tmp/eval_logs')
    logger = setup_logging(log_dir)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    dataset_config = config['dataset']
    # Allow command-line override of dataset
    dataset_name = args.dataset if args.dataset else dataset_config['name']
    cache_dir = dataset_config.get('cache_dir', None)
    
    # Load from evaluation dataset registry
    if dataset_name.lower() in EVAL_DATASETS:
        if is_main_process():
            logger.info(f"Loading {dataset_name.upper()} evaluation dataset...")
        test_data = load_eval_dataset(dataset_name, cache_dir)
    else:
        if is_main_process():
            logger.error(f"Unknown evaluation dataset: {dataset_name}")
            logger.info(f"Supported evaluation datasets: {list(EVAL_DATASETS.keys())}")
        sys.exit(1)

    if is_main_process():
        logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize reward model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get reward model parameters from config (if available)
    training_config = config.get('training', {})
    use_format_reward = training_config.get('use_format_reward', False)
    reward_model = create_reward_model(
        dataset_name, 
        reward_type='accuracy',
        use_format_reward=use_format_reward,
    )
    
    # Get evaluation config
    eval_config = config.get('evaluation', {})
    
    # Determine checkpoints to test
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        if is_main_process():
            logger.error(f"Checkpoint {checkpoint_path} does not exist!")
        sys.exit(1)
    
    if is_main_process():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Model: {checkpoint_path}")
        logger.info(f"{'='*60}")
    
    final_results = evaluate_model_vllm(
        str(checkpoint_path),
        test_data,
        reward_model,
        device,
        eval_config,
        num_gpus=args.num_gpus,
        dataset_name=dataset_name,
    )
    
    if is_main_process():
        logger.info(f"Results: {final_results}")
    
    # Save results if output file specified
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    if is_main_process():
        logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()
