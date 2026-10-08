"""
Model checkpoint utilities
"""

import os
import logging
from utils.experiment import is_main_process


def save_model_checkpoint(
    model,
    tokenizer,
    checkpoint_dir: str,
    logger: logging.Logger
) -> None:
    """Save model and tokenizer checkpoint with distributed training support.
    
    Handles Accelerate wrapper unwrapping and multi-process synchronization.
    Only the main process saves the checkpoint.
    
    Args:
        model: Model to save (may be wrapped by Accelerate)
        tokenizer: Tokenizer to save
        checkpoint_dir: Directory to save checkpoint
        logger: Logger instance
    """
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        
        # Unwrap model from Accelerate wrapper
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        
        # Synchronize all processes before saving
        accelerator.wait_for_everyone()
        
        # Only main process saves
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {checkpoint_dir}...")
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Checkpoint saved successfully")
        
        # Wait for saving to complete
        accelerator.wait_for_everyone()
    except ImportError:
        # Fallback without Accelerator
        if is_main_process():
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {checkpoint_dir}...")
            model.eval()
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Checkpoint saved successfully")
