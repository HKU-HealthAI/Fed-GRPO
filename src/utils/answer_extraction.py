"""
Unified Answer Extraction Utilities for Math Datasets

Supports:
- R1-style <answer>...</answer> tag extraction
"""

import re
import logging
from typing import Optional, Callable
from utils.experiment import is_main_process

logger = logging.getLogger(__name__)


def extract_boxed_content(text: str, use_last: bool = True) -> str:
    """
    Extract content from \\boxed{} with balanced bracket matching.
    Handles nested braces like \\boxed{\\frac{1}{2}}.

    Args:
        text: Input text containing \\boxed{...}
        use_last: If True, extract from the LAST \\boxed{} (for final answers).
                  If False, extract from the FIRST one.

    Returns:
        Content inside the \\boxed{}, or empty string if not found
    """
    # Find ALL \boxed{ positions
    boxed_pattern = r'\\boxed\{'
    matches = list(re.finditer(boxed_pattern, text))
    if not matches:
        return ""

    # Select which match to use
    target_match = matches[-1] if use_last else matches[0]

    # Start after \boxed{
    start_idx = target_match.end()

    # Use balanced bracket matching
    depth = 1
    idx = start_idx
    while idx < len(text) and depth > 0:
        if text[idx] == '{':
            depth += 1
        elif text[idx] == '}':
            depth -= 1
        idx += 1

    if depth == 0:
        # Extract content (excluding the final })
        return text[start_idx:idx-1]

    return ""


def extract_answer_tag_content(text: str) -> str:
    """
    Extract content from <answer>...</answer> tags (R1-style format).
    
    Args:
        text: Model output text
        
    Returns:
        Content inside <answer> tags, or empty string if not found
    """
    # Match <answer>...</answer> pattern
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_math_answer(model_output: str) -> Optional[str]:
    """
    Extract answer from model output for MATH dataset.
    
    Supports complex LaTeX expressions with normalization.
    - R1-style: <answer>expression</answer>
    - Boxed inside answer: <answer>\\boxed{expression}</answer>
    
    Args:
        model_output: Model generated response
        
    Returns:
        Extracted and normalized answer string, or None if not found
    """
    # First try to extract from <answer> tag
    answer_content = extract_answer_tag_content(model_output)
    
    if answer_content:
        content = answer_content.strip()
        
        # Check if content contains \boxed{} inside answer tag
        boxed_in_answer = extract_boxed_content(content)
        if boxed_in_answer:
            content = boxed_in_answer.strip()
        
        # Normalize the answer (LaTeX-aware)
        return _normalize_math_answer(content)
    
    # Fallback: extract from \boxed{} directly (for truncated completions
    # that never reached <answer> tag but may contain \boxed{answer})
    boxed_content = extract_boxed_content(model_output)
    if boxed_content:
        return _normalize_math_answer(boxed_content.strip())
    
    return None


def _normalize_math_answer(answer: str) -> str:
    """
    Normalize a mathematical answer string.
    Handles LaTeX expressions and numerical values.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    answer = str(answer).strip()
    
    # Remove $ symbols
    answer = answer.replace("$", "")

    # Remove LaTeX inline/display wrappers \(...\) and \[...\]
    answer = answer.replace("\\(", "").replace("\\)", "")
    answer = answer.replace("\\[", "").replace("\\]", "")
    
    # Normalize \tfrac and \dfrac to \frac
    answer = answer.replace("\\tfrac", "\\frac")
    answer = answer.replace("\\dfrac", "\\frac")
    
    # Normalize \frac shorthand: \fracXY -> \frac{X}{Y} (single-char args without braces)
    # e.g. \frac16 -> \frac{1}{6}, \fracab -> \frac{a}{b}
    answer = re.sub(r'\\frac\s*([^{\s\\])\s*([^{\s\\])', r'\\frac{\1}{\2}', answer)
    # Also handle \fracX{Y} -> \frac{X}{Y} (first arg unbraced, second braced)
    answer = re.sub(r'\\frac\s*([^{\s\\])\s*(\{)', r'\\frac{\1}\2', answer)
    
    # Remove \left and \right
    answer = answer.replace("\\left", "").replace("\\right", "")
    
    # Remove \text{} wrappers
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    
    # Normalize 100,000 -> 100000
    if answer.replace(",", "").replace(".", "").replace("-", "").isdigit():
        answer = answer.replace(",", "")
    
    return answer.strip()


# ===========================
# Dataset-specific Extractors
# For evaluation purposes
# ===========================

# Dataset -> Extractor function mapping
# All math datasets use the unified extract_math_answer function
DATASET_EXTRACTORS = {
    # All datasets use math-style extraction (handles both numerical and symbolic)
    'gsm8k': extract_math_answer,
    'amc': extract_math_answer,
    'math': extract_math_answer,
    'math500': extract_math_answer,
    'aime2024': extract_math_answer,
    'aime2025': extract_math_answer,
}


def get_answer_extractor(dataset_name: str) -> Callable[[str], Optional[str]]:
    """
    Get the appropriate answer extractor function for a dataset.
    
    Args:
        dataset_name: Dataset name (gsm8k, math, math500, aime2024, aime2025, amc)
        
    Returns:
        Answer extractor function (text) -> answer_string
    """
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in DATASET_EXTRACTORS:
        return DATASET_EXTRACTORS[dataset_name_lower]
    else:
        # Default to math-style extraction for unknown datasets
        return extract_math_answer


def extract_answer(text: str, dataset_name: str = None) -> str:
    """
    Universal answer extraction function.
    
    When dataset_name is provided, uses the dataset-specific extractor.
    Otherwise defaults to math-style extraction.
    
    Args:
        text: Model output text
        dataset_name: Optional dataset name for specific extraction
        
    Returns:
        Extracted answer string
    """
    if dataset_name:
        extractor = get_answer_extractor(dataset_name)
        result = extractor(text)
        return result if result else ""
    else:
        # Default to math extraction
        result = extract_math_answer(text)
        return result if result else ""
    