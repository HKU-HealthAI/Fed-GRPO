"""
Reward functions for MATH dataset.

Uses Math-Verify for robust mathematical answer verification.

Model output format (R1-style): "<think>reasoning</think> <answer>expression</answer>"
"""

import re
import logging
from fractions import Fraction

from utils.answer_extraction import extract_math_answer, _normalize_math_answer
from utils.experiment import is_main_process

logger = logging.getLogger(__name__)


# ============================================================================
# Math-Verify Import
# ============================================================================

_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import parse as mv_parse, verify as mv_verify
    from math_verify import LatexExtractionConfig, ExprExtractionConfig
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    if is_main_process():
        logger.warning(
            "math-verify not available. Install: pip install math-verify[antlr4_13_2]. "
            "Falling back to simple string comparison."
        )


# ============================================================================
# Safety Check
# ============================================================================

def repeatness(s: str) -> bool:
    """Check if a string is repetitive (reward hacking indicator).

    Reference: sail-sg/understand-r1-zero math_grader.py repeatness()
    """
    if len(s) < 10:
        return False
    for pattern_len in range(1, min(20, len(s) // 3)):
        pattern = s[:pattern_len]
        count = s.count(pattern)
        if count >= len(s) // pattern_len * 0.8:
            return True
    return False


# ============================================================================
# R1 Format Checking and Reward
# ============================================================================

def check_r1_format(text: str) -> dict:
    """Check if text follows R1 format: <think>...</think> <answer>...</answer>."""
    result = {
        'has_think': False,
        'has_answer': False,
        'has_full_format': False,
        'think_content': '',
        'answer_content': ''
    }

    text_lower = text.lower()
    think_open_pos = 0  # we prepend <think> in the prompt
    think_close_pos = text_lower.find('</think>')

    if think_open_pos >= 0 and think_close_pos >= 0 and think_open_pos < think_close_pos:
        result['has_think'] = True
        result['think_content'] = text[think_open_pos + 7:think_close_pos].strip()

    answer_open_pos = text_lower.find('<answer>')
    answer_close_pos = text_lower.find('</answer>')

    if answer_open_pos >= 0 and answer_close_pos >= 0 and answer_open_pos < answer_close_pos:
        result['has_answer'] = True
        result['answer_content'] = text[answer_open_pos + 8:answer_close_pos].strip()

    if result['has_think'] and result['has_answer']:
        if think_close_pos < answer_open_pos:
            result['has_full_format'] = True

    return result


def format_reward(text: str, strict: bool = True) -> float:
    """Compute format reward for R1-style output.

    Returns:
        strict=True:  1.0 for perfect format, 0.0 otherwise
        strict=False: 0.0~1.0 partial reward
    """
    fmt = check_r1_format(text)

    if strict:
        return 1.0 if fmt['has_full_format'] else 0.0

    if fmt['has_full_format']:
        return 1.0
    if fmt['has_think'] and fmt['has_answer']:
        return 0.7
    if fmt['has_think']:
        return 0.5
    if fmt['has_answer']:
        return 0.3
    return 0.0


# ============================================================================
# Simple String Fallback (when Math-Verify is unavailable)
# ============================================================================

_RE_TEXT_WRAPPER = re.compile(r'\\(?:text|textbf|mathrm)\{([^}]*)\}')
_RE_EQ_SPLIT = re.compile(r'(?<!\\)=')
_RE_NUMERIC_UNIT = re.compile(
    r'^\s*([+-]?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*'
    r'(?:\\mbox\{[^}]*\}|\\text\{[^}]*\}|\\mathrm\{[^}]*\}|%|\\%|[A-Za-z\\^{}._\s]+)\s*$'
)
_RE_PERCENT = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+))\s*(?:%|\\%)\s*$')
_RE_DECIMAL = re.compile(r'^\s*[+-]?\d*\.\d+\s*$')


def _simple_normalize(answer: str) -> str:
    """Minimal normalization for string fallback comparison."""
    if not answer:
        return ""
    s = str(answer).strip()
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    s = _RE_TEXT_WRAPPER.sub(r'\1', s)
    s = s.replace(" ", "").replace("\\,", "").replace("\\!", "")
    if s.replace(",", "").replace(".", "").replace("-", "").isdigit():
        s = s.replace(",", "")
    return s


def _candidate_predictions(predicted: str) -> list:
    """Generate candidate answers from a prediction.

    Handles equality chains like "a=b=c" by including the last segment as a candidate.
    Keeps the original prediction as the first candidate.
    """
    candidates = []
    if not predicted:
        return candidates

    raw = predicted.strip()
    if raw:
        candidates.append(raw)

    # Unwrap simple text wrappers like \text{...}
    m = _RE_TEXT_WRAPPER.fullmatch(raw)
    if m:
        inner = m.group(1).strip()
        if inner and inner not in candidates:
            candidates.append(inner)

    # Split on unescaped '=' and add the last segment if different
    parts = [p.strip() for p in _RE_EQ_SPLIT.split(raw) if p.strip()]
    if len(parts) >= 2:
        last_part = parts[-1]
        if last_part and last_part not in candidates:
            candidates.append(last_part)

    # Add numeric-only candidate when units/percent are present
    more = []
    for cand in candidates:
        m = _RE_NUMERIC_UNIT.match(cand)
        if m:
            numeric = m.group(1).replace(" ", "")
            if numeric and numeric not in candidates and numeric not in more:
                more.append(numeric)

        # Decimal -> fraction candidate (e.g., 7.5 -> 15/2)
        if _RE_DECIMAL.match(cand):
            try:
                frac = Fraction(cand.strip()).limit_denominator()
                frac_str = f"{frac.numerator}/{frac.denominator}"
                if frac_str not in candidates and frac_str not in more:
                    more.append(frac_str)
            except Exception:
                pass

        # LaTeX \frac{a}{b} -> a/b candidate (e.g., \frac{11}{5} -> 11/5)
        frac_m = re.match(r'^(-?)\\frac\{([^}]+)\}\{([^}]+)\}$', cand.strip())
        if frac_m:
            sign = frac_m.group(1)
            plain_frac = f"{sign}{frac_m.group(2)}/{frac_m.group(3)}"
            if plain_frac not in candidates and plain_frac not in more:
                more.append(plain_frac)

        # If percent, also add numeric/100 equivalent
        pm = _RE_PERCENT.match(cand)
        if pm:
            numeric = pm.group(1).replace(" ", "")
            if numeric:
                decimal_expr = f"({numeric})/100"
                if decimal_expr not in candidates and decimal_expr not in more:
                    more.append(decimal_expr)
    candidates.extend(more)

    return candidates


# ============================================================================
# Numerical Comparison Helper
# ============================================================================

def _try_numeric_value(s: str):
    """Try to evaluate a math string as a numeric value.

    Supports plain numbers, simple fractions (a/b), and LaTeX fractions (\\frac{a}{b}).

    Returns:
        float value if parseable, None otherwise
    """
    s = s.strip()
    if not s:
        return None

    # Direct number (e.g., 2.2, -3, 42)
    try:
        return float(s)
    except ValueError:
        pass

    # Handle optional leading negative sign before \frac
    negative = False
    core = s
    if s.startswith('-'):
        negative = True
        core = s[1:].strip()

    # Simple fraction a/b (e.g., 11/5, -1/2)
    m = re.match(r'^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$', core)
    if m:
        try:
            val = float(m.group(1)) / float(m.group(2))
            return -val if negative else val
        except (ValueError, ZeroDivisionError):
            pass

    # LaTeX \frac{a}{b} (e.g., \frac{11}{5})
    m = re.match(r'^\\frac\{([^}]+)\}\{([^}]+)\}$', core)
    if m:
        try:
            val = float(m.group(1)) / float(m.group(2))
            return -val if negative else val
        except (ValueError, ZeroDivisionError):
            pass

    return None


# ============================================================================
# Core Reward Function
# ============================================================================

def math_reward_function(model_output: str, ground_truth: str) -> float:
    """
    Binary reward using Math-Verify:
      R(q, o) = 1  if o contains the correct final answer
      R(q, o) = 0  otherwise

    Pipeline:
      1. Extract answer from <answer> tag (or \\boxed{} fallback)
      2. Reject repetitive garbage outputs
      3. Use Math-Verify parse() + verify() for robust comparison
      4. Fall back to string comparison if Math-Verify fails/unavailable

    Args:
        model_output: Model generated response (with <think>/<answer> tags)
        ground_truth: Correct answer from dataset (may be LaTeX)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # Step 0: Normalize ground truth (e.g., \frac16 -> \frac{1}{6})
    ground_truth = _normalize_math_answer(ground_truth)

    # Step 1: Extract answer from model output
    predicted = extract_math_answer(model_output)
    if not predicted:
        return 0.0

    candidates = _candidate_predictions(predicted)
    if not candidates:
        return 0.0

    # Step 2: Reject repetitive/garbage outputs
    if len(predicted) > 128 and repeatness(predicted):
        return 0.0

    # Step 3: Math-Verify comparison
    if _MATH_VERIFY_AVAILABLE:
        try:
            ref_candidates = _candidate_predictions(ground_truth)
            if not ref_candidates:
                ref_candidates = [ground_truth]

            for ref_cand in ref_candidates:
                ref_str = ref_cand.replace("\n", "")
                ref_str = ref_str.replace("\\(", "").replace("\\)", "")
                ref_str = ref_str.replace("\\[", "").replace("\\]", "")
                if "$" not in ref_str:
                    ref_str = f"${ref_str}$"

                ref_parsed = mv_parse(
                    ref_str,
                    extraction_config=[
                        LatexExtractionConfig(
                            boxed="all",
                            basic_latex=True,
                            units=True,
                            malformed_operators=False,
                            equations=False,
                        ),
                        ExprExtractionConfig(),
                    ],
                    fallback_mode="no_fallback",
                    extraction_mode=["first_match"],
                    parsing_timeout=1,
                )

                for cand in candidates:
                    pred_str = cand.replace("\n", "")
                    pred_str = pred_str.replace("\\(", "").replace("\\)", "")
                    pred_str = pred_str.replace("\\[", "").replace("\\]", "")
                    if "$" not in pred_str:
                        pred_str = f"${pred_str}$"

                    pred_parsed = mv_parse(
                        pred_str,
                        extraction_config=[
                            LatexExtractionConfig(
                                boxed="all",
                                boxed_match_priority=0,
                                basic_latex=True,
                                units=True,
                                malformed_operators=False,
                                equations=False,
                            ),
                            ExprExtractionConfig(),
                        ],
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    )

                    if mv_verify(ref_parsed, pred_parsed, timeout_seconds=2):
                        return 1.0
        except Exception:
            # Math-Verify failed — fall through to string comparison
            pass

    # Step 4: Simple string fallback
    ref_candidates = _candidate_predictions(ground_truth)
    if not ref_candidates:
        ref_candidates = [ground_truth]
    ref_norms = {_simple_normalize(r) for r in ref_candidates if r is not None}
    for cand in candidates:
        if _simple_normalize(cand) in ref_norms:
            return 1.0

    # Step 5: Numerical equivalence fallback (e.g., 2.2 == \frac{11}{5})
    for cand in candidates:
        pred_val = _try_numeric_value(_simple_normalize(cand))
        if pred_val is not None:
            for ref_cand in ref_candidates:
                ref_val = _try_numeric_value(_simple_normalize(ref_cand))
                if ref_val is not None and abs(pred_val - ref_val) < 1e-9:
                    return 1.0

    return 0.0


def math_reward_with_format(model_output: str, ground_truth: str) -> float:
    """MATH reward + R1 format reward. Returns 0.0~2.0."""
    return math_reward_function(model_output, ground_truth) + format_reward(model_output, strict=True)


__all__ = [
    "check_r1_format",
    "format_reward",
    "math_reward_function",
    "math_reward_with_format",
]
