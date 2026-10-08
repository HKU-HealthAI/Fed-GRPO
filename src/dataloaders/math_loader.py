"""
MATH Dataset Loader for Training and Evaluation

Training: HuggingFaceH4/MATH (7 subject subsets, train split)
Evaluation:
  - MATH-500: HuggingFaceH4/MATH-500
  - GSM8K: openai/gsm8k (test split)
  - AIME 2024: HuggingFaceH4/aime_2024
  - AIME 2025: yentinglin/aime_2025  
  - AMC: math-ai/amc23
"""

import re
import logging
from typing import List, Dict, Optional
from datasets import load_dataset
from .base_loader import BaseDataset
from utils import extract_boxed_content
from utils.experiment import is_main_process

logger = logging.getLogger(__name__)


# ===========================
# Evaluation Dataset Registry
# ===========================

EVAL_DATASETS = {
    'math500': {
        'path': 'HuggingFaceH4/MATH-500',
        'split': 'test',
        'problem_field': 'problem',
        'answer_field': 'answer',
        'solution_field': 'solution',
        'template': 'math_symbolic',  # \boxed{} style
    },
    'gsm8k': {
        'path': 'openai/gsm8k',
        'name': 'main',
        'split': 'test',
        'problem_field': 'question',
        'answer_field': 'answer',  # Needs extraction: #### number
        'solution_field': None,
        'template': 'math_symbolic',
        'answer_extractor': 'gsm8k',
    },
    'aime2024': {
        'path': 'HuggingFaceH4/aime_2024',
        'split': 'train',  # Only has train split
        'problem_field': 'problem',
        'answer_field': 'answer',
        'solution_field': 'solution',
        'template': 'math_symbolic',
    },
    'aime2025': {
        'path': 'yentinglin/aime_2025',
        'split': 'train',  # Only has train split
        'problem_field': 'problem',
        'answer_field': 'answer',
        'solution_field': 'solution',
        'template': 'math_symbolic',
    },
    'amc': {
        'path': 'math-ai/amc23',
        'split': 'test',
        'problem_field': 'question',
        'answer_field': 'answer',
        'solution_field': None,
        'template': 'math_symbolic',
    },
}


class MATHDataset(BaseDataset):
    """MATH Competition-level Mathematics Dataset"""
    
    SUBJECTS = {
        0: "algebra",
        1: "counting_and_probability",
        2: "geometry",
        3: "intermediate_algebra",
        4: "number_theory",
        5: "prealgebra",
        6: "precalculus"
    }
    
    def __init__(self, num_clients: int = 5, alpha: float = 0.1, cache_dir: str = None, hard_only: bool = False):
        self.hard_only = hard_only
        super().__init__(num_clients, alpha, cache_dir)
        self.load_data()
    
    # The 7 subject subsets from HuggingFaceH4/MATH (excluding "default" which duplicates "precalculus")
    MATH_SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    
    def load_data(self) -> None:
        """Load MATH training dataset from HuggingFaceH4/MATH (7 subject subsets, train split).
        
        When hard_only=True, only Level 3-5 problems are kept.
        """
        if is_main_process():
            filter_desc = " (hard_only: Level 3-5)" if self.hard_only else " (all levels)"
            logger.info(f"Loading MATH training dataset from HuggingFaceH4/MATH{filter_desc} ...")
        
        ALLOWED_LEVELS = {3, 4, 5}
        total_skipped = 0
        subset_counts = {}
        
        try:
            for subset_name in self.MATH_SUBSETS:
                if is_main_process():
                    logger.info(f"  Loading subset: {subset_name} (train split)...")
                
                dataset = load_dataset(
                    "HuggingFaceH4/MATH",
                    name=subset_name,
                    split="train",
                    cache_dir=self.cache_dir,
                )
                
                subset_added = 0
                subset_skipped = 0
                
                for item in dataset:
                    problem = item["problem"]
                    solution = item["solution"]
                    level = item.get("level", 1)

                    # Filter by difficulty level when hard_only is enabled
                    if self.hard_only:
                        parsed_level = level
                        if isinstance(level, str):
                            match = re.search(r'(\d+)', level)
                            parsed_level = int(match.group(1)) if match else 0
                        if parsed_level not in ALLOWED_LEVELS:
                            subset_skipped += 1
                            continue

                    # HuggingFaceH4/MATH uses "type" field for subject
                    raw_subject = item.get("type") or item.get("subject", subset_name)
                    subject = raw_subject if raw_subject is not None else subset_name

                    # The final answer is boxed in the LaTeX solution; extract it.
                    answer = item.get("answer")
                    if answer is None:
                        answer = extract_boxed_content(solution) or ""
                    
                    self.data.append({
                        "prompt": self.templates.math_symbolic(problem),
                        "answer": answer,
                        "full_solution": solution,
                        "level": level,
                        "subject": subject,
                        "subject_id": self._get_subject_id(subject)
                    })
                    subset_added += 1
                
                subset_counts[subset_name] = subset_added
                total_skipped += subset_skipped
            
            if is_main_process():
                if self.hard_only:
                    logger.info(f"Loaded {len(self.data)} training samples from HuggingFaceH4/MATH "
                                f"(skipped {total_skipped} Level 1-2 samples)")
                else:
                    logger.info(f"Loaded {len(self.data)} training samples from HuggingFaceH4/MATH")
                logger.info(f"Per-subset counts: {subset_counts}")
        
        except Exception as e:
            if is_main_process():
                logger.error(f"Failed to load MATH: {e}")
            raise RuntimeError(f"Failed to load MATH dataset: {e}")
    
    def _get_subject_id(self, subject: str) -> int:
        """Map subject name to ID"""
        for sid, name in self.SUBJECTS.items():
            # Normalize common variants from HF datasets:
            # - spaces -> underscores
            # - '&' -> 'and'
            normalized = subject.lower().replace('&', 'and').replace(' ', '_')
            if name.lower() == normalized:
                return sid
        return 0  # Default to algebra
    
    def _get_difficulty_group(self, level) -> int:
        """Map difficulty level to client index.
        
        When hard_only=True (Level 3-5 only, 3 clients):
          Level 3 -> client 0
          Level 4 -> client 1
          Level 5 -> client 2
        
        When hard_only=False (all levels, 5 clients):
          Level 1 -> client 0
          Level 2 -> client 1
          Level 3 -> client 2
          Level 4 -> client 3
          Level 5 -> client 4
        
        Args:
            level: Difficulty level (1-5 or string like "Level 3")
        
        Returns:
            Client index
        """
        # Handle string format like "Level 3"
        if isinstance(level, str):
            match = re.search(r'(\d+)', level)
            level = int(match.group(1)) if match else 1
        
        if self.hard_only:
            # Level 3-5 -> client 0-2
            level = max(3, min(5, level))
            return level - 3
        else:
            # Level 1-5 -> client 0-4
            level = max(1, min(5, level))
            return level - 1
    
    def split_data(self, method: str = "dirichlet") -> None:
        """
        Split data for each client.
        
        Args:
            method: Split method
                - "dirichlet": Dirichlet distribution split by subject (default, recommended)
                - "subject": Hard split by subject (FIXED 7 clients)
                - "difficulty": Hard split by difficulty level (3 clients for hard_only, 5 otherwise)
                - "random": Random IID split
        """
        if is_main_process():
            logger.info(f"Splitting MATH data using method: {method}")
        
        if method == "dirichlet":
            # Use Dirichlet distribution by subject
            labels = [item['subject_id'] for item in self.data]
            client_indices = self.dirichlet_split(labels)
        
        elif method == "subject":
            # Hard split by subject - FIXED 7 clients
            if self.num_clients != 7:
                if is_main_process():
                    logger.warning(f"subject split requires exactly 7 clients, but got {self.num_clients}. Forcing to 7 clients.")
                self.num_clients = 7
                self.client_data = [[] for _ in range(7)]
            
            client_indices = [[] for _ in range(7)]
            for idx, item in enumerate(self.data):
                subject_id = item['subject_id']
                client_indices[subject_id].append(idx)
            
            if is_main_process():
                subject_counts = {self.SUBJECTS[i]: len(client_indices[i]) for i in range(7)}
                logger.info(f"Subject split distribution: {subject_counts}")
        
        elif method == "difficulty":
            # Hard split by difficulty level
            # hard_only=True: 3 clients (Level 3/4/5), hard_only=False: 5 clients (Level 1-5)
            expected_clients = 3 if self.hard_only else 5
            if self.num_clients != expected_clients:
                if is_main_process():
                    logger.warning(f"difficulty split requires exactly {expected_clients} clients "
                                   f"(hard_only={self.hard_only}), but got {self.num_clients}. "
                                   f"Forcing to {expected_clients} clients.")
                self.num_clients = expected_clients
                self.client_data = [[] for _ in range(expected_clients)]
            
            client_indices = [[] for _ in range(expected_clients)]
            for idx, item in enumerate(self.data):
                difficulty_group = self._get_difficulty_group(item['level'])
                client_indices[difficulty_group].append(idx)
            
            if is_main_process():
                start_level = 3 if self.hard_only else 1
                level_counts = {f"Level {i + start_level}": len(client_indices[i]) for i in range(expected_clients)}
                logger.info(f"Difficulty split distribution: {level_counts}")
        
        else:  # random
            client_indices = self.random_split()
        
        # Allocate data using base class method
        self._allocate_data(client_indices)


# ===========================
# Evaluation Dataset Loaders
# ===========================

def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract final numerical answer from GSM8K answer text (#### number)."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: find last number
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')
    return "0"


def load_eval_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Load an evaluation dataset.
    
    Args:
        dataset_name: One of 'math500', 'gsm8k', 'aime2024', 'aime2025', 'amc'
        cache_dir: Dataset cache directory
        
    Returns:
        List of {'prompt': str, 'answer': str} dicts
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower not in EVAL_DATASETS:
        raise ValueError(
            f"Unknown evaluation dataset: {dataset_name}. "
            f"Supported: {list(EVAL_DATASETS.keys())}"
        )
    
    config = EVAL_DATASETS[dataset_name_lower]
    
    if is_main_process():
        logger.info(f"Loading evaluation dataset: {dataset_name} from {config['path']}...")
    
    try:
        # Load dataset
        load_kwargs = {
            'path': config['path'],
            'split': config['split'],
            'cache_dir': cache_dir,
        }
        if 'name' in config:
            load_kwargs['name'] = config['name']
        
        dataset = load_dataset(**load_kwargs)
        
        # Get template function
        from .base_loader import PromptTemplates
        template_func = getattr(PromptTemplates, config['template'])
        
        # Process items
        eval_data = []
        for item in dataset:
            problem = item.get(config['problem_field'], '')
            raw_answer = item.get(config['answer_field'], '')
            
            # Extract answer if needed (e.g., GSM8K has #### format)
            if config.get('answer_extractor') == 'gsm8k':
                answer = _extract_gsm8k_answer(raw_answer)
            else:
                answer = str(raw_answer).strip()
            
            eval_data.append({
                'prompt': template_func(problem),
                'answer': answer,
            })
        
        if is_main_process():
            logger.info(f"Loaded {len(eval_data)} samples from {dataset_name}")
        
        return eval_data
        
    except Exception as e:
        if is_main_process():
            logger.error(f"Failed to load {dataset_name}: {e}")
        raise RuntimeError(f"Failed to load evaluation dataset {dataset_name}: {e}")


def load_all_eval_datasets(cache_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Load all evaluation datasets.
    
    Args:
        cache_dir: Dataset cache directory
        
    Returns:
        Dict mapping dataset name to list of samples
    """
    all_datasets = {}
    for name in EVAL_DATASETS.keys():
        try:
            all_datasets[name] = load_eval_dataset(name, cache_dir)
        except Exception as e:
            if is_main_process():
                logger.warning(f"Skipping {name}: {e}")
    return all_datasets
