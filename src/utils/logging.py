"""
Logging and metrics utilities
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from .experiment import is_main_process


class TrainingMetricsLogger:
    """
    Unified training metrics logger for all training methods.

    Features:
    - Consistent metrics logging across Fed-GRPO, FedAvg, Local-only, Centralized
    - JSON output for easy parsing and plotting
    - Round/step summary statistics
    - Progress tracking with ETA estimation
    - Best metrics tracking
    """

    def __init__(
        self,
        log_dir: str,
        method: str,
        num_rounds: Optional[int] = None,
        num_clients: Optional[int] = None,
        max_steps: Optional[int] = None
    ):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save logs
            method: Training method (fed_grpo, fedavg, local_only, centralized)
            num_rounds: Total number of rounds (for federated methods)
            num_clients: Number of clients (for federated methods)
            max_steps: Total training steps (for non-federated methods)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.max_steps = max_steps

        # Metrics storage
        self.round_metrics: List[Dict] = []
        self.client_metrics: List[List[Dict]] = []  # Per-round client metrics
        self.step_metrics: List[Dict] = []

        # Best metrics tracking
        self.best_reward = float('-inf')
        self.best_reward_round = -1
        self.lowest_loss = float('inf')
        self.lowest_loss_round = -1

        # Timing
        self.start_time = time.time()
        self.round_start_time = None

        # Logger
        self.logger = logging.getLogger(__name__)

        # JSON log file
        self.metrics_file = self.log_dir / 'metrics.jsonl'

    def start_round(self, round_idx: int):
        """Mark the start of a training round."""
        self.round_start_time = time.time()
        self.current_round = round_idx
        self.current_round_client_metrics = []

    def log_client(self, client_id: int, metrics: Dict[str, Any]):
        """Log metrics for a single client."""
        client_entry = {
            'round': self.current_round,
            'client_id': client_id,
            'timestamp': datetime.now().isoformat(),
            **self._extract_metrics(metrics)
        }
        self.current_round_client_metrics.append(client_entry)

        # Log to console with consistent format
        if is_main_process():
            self.logger.info(
                f"Client {client_id}: "
                f"loss={metrics.get('loss', 0):.4f}, "
                f"reward={metrics.get('avg_reward', 0):.4f}, "
                f"reward_std={metrics.get('reward_std', 0):.4f}, "
                f"kl={metrics.get('avg_kl', 0):.4f}, "
                f"grad_norm={metrics.get('avg_grad_norm', 0):.4f}"
            )

    def end_round(self, aggregation_stats: Optional[Dict] = None):
        """
        Mark the end of a training round and compute summary statistics.

        Args:
            aggregation_stats: Optional stats from server aggregation
        """
        round_time = time.time() - self.round_start_time
        elapsed_time = time.time() - self.start_time

        # Compute round summary
        if self.current_round_client_metrics:
            avg_loss = np.mean([m['loss'] for m in self.current_round_client_metrics])
            avg_reward = np.mean([m['avg_reward'] for m in self.current_round_client_metrics])
            avg_reward_std = np.mean([m['reward_std'] for m in self.current_round_client_metrics])
            avg_kl = np.mean([m['avg_kl'] for m in self.current_round_client_metrics])
            avg_grad_norm = np.mean([m['avg_grad_norm'] for m in self.current_round_client_metrics])
            avg_response_length = np.mean([m['avg_response_length'] for m in self.current_round_client_metrics])
        else:
            avg_loss = avg_reward = avg_reward_std = avg_kl = avg_grad_norm = avg_response_length = 0

        # Update best metrics
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_reward_round = self.current_round
        if avg_loss < self.lowest_loss:
            self.lowest_loss = avg_loss
            self.lowest_loss_round = self.current_round

        # ETA calculation
        if self.num_rounds and self.current_round > 0:
            avg_round_time = elapsed_time / (self.current_round + 1)
            remaining_rounds = self.num_rounds - self.current_round - 1
            eta_seconds = avg_round_time * remaining_rounds
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "N/A"

        # Round summary entry
        round_summary = {
            'round': self.current_round,
            'timestamp': datetime.now().isoformat(),
            'round_time': round_time,
            'elapsed_time': elapsed_time,
            'avg_loss': float(avg_loss),
            'avg_reward': float(avg_reward),
            'avg_reward_std': float(avg_reward_std),
            'avg_kl': float(avg_kl),
            'avg_grad_norm': float(avg_grad_norm),
            'avg_response_length': float(avg_response_length),
            'best_reward': float(self.best_reward),
            'best_reward_round': self.best_reward_round,
            'lowest_loss': float(self.lowest_loss),
            'lowest_loss_round': self.lowest_loss_round,
        }

        if aggregation_stats:
            round_summary['aggregation_weights'] = aggregation_stats.get('weights', [])

        self.round_metrics.append(round_summary)
        self.client_metrics.append(self.current_round_client_metrics)

        # Write to JSONL file
        self._write_jsonl({'type': 'round_summary', **round_summary})
        for client_entry in self.current_round_client_metrics:
            self._write_jsonl({'type': 'client', **client_entry})

        # Log round summary
        if is_main_process():
            progress = f"[{self.current_round + 1}/{self.num_rounds}]" if self.num_rounds else f"[Round {self.current_round + 1}]"
            self.logger.info(
                f"\n{'─'*60}\n"
                f"Round {progress} Summary | Time: {round_time:.1f}s | ETA: {eta_str}\n"
                f"  Loss: {avg_loss:.4f} (best: {self.lowest_loss:.4f} @ R{self.lowest_loss_round + 1})\n"
                f"  Reward: {avg_reward:.4f} (best: {self.best_reward:.4f} @ R{self.best_reward_round + 1})\n"
                f"  Reward Std: {avg_reward_std:.4f} | KL: {avg_kl:.4f} | Grad Norm: {avg_grad_norm:.4f}\n"
                f"{'─'*60}"
            )

    def log_step(self, step: int, metrics: Dict[str, Any], client_name: str = ""):
        """Log metrics for a single training step (for non-federated methods)."""
        step_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'client_name': client_name,
            **self._extract_metrics(metrics)
        }
        self.step_metrics.append(step_entry)
        self._write_jsonl({'type': 'step', **step_entry})

    def log_training_complete(self, final_metrics: Dict[str, Any], client_name: str = ""):
        """Log final training completion metrics."""
        elapsed_time = time.time() - self.start_time

        # Update best metrics
        avg_reward = final_metrics.get('avg_reward', 0)
        avg_loss = final_metrics.get('loss', 0)
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
        if avg_loss < self.lowest_loss:
            self.lowest_loss = avg_loss

        completion_entry = {
            'type': 'training_complete',
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'method': self.method,
            'client_name': client_name,
            'final_loss': float(avg_loss),
            'final_reward': float(avg_reward),
            'best_reward': float(self.best_reward),
            'lowest_loss': float(self.lowest_loss),
            **self._extract_metrics(final_metrics)
        }
        self._write_jsonl(completion_entry)

        # Log completion summary
        if is_main_process():
            time_str = str(timedelta(seconds=int(elapsed_time)))
            name_str = f" ({client_name})" if client_name else ""
            self.logger.info(
                f"\n{'='*60}\n"
                f"Training Complete{name_str} | Total Time: {time_str}\n"
                f"  Final Loss: {avg_loss:.4f} | Final Reward: {avg_reward:.4f}\n"
                f"  Best Reward: {self.best_reward:.4f} | Lowest Loss: {self.lowest_loss:.4f}\n"
                f"  Reward Std: {final_metrics.get('reward_std', 0):.4f}\n"
                f"  Grad Norm: {final_metrics.get('avg_grad_norm', 0):.4f}\n"
                f"{'='*60}"
            )

    def save_summary(self):
        """Save final summary to JSON file."""
        summary = {
            'method': self.method,
            'num_rounds': self.num_rounds,
            'num_clients': self.num_clients,
            'max_steps': self.max_steps,
            'total_time': time.time() - self.start_time,
            'best_reward': float(self.best_reward),
            'best_reward_round': self.best_reward_round,
            'lowest_loss': float(self.lowest_loss),
            'lowest_loss_round': self.lowest_loss_round,
            'round_metrics': self.round_metrics,
        }

        summary_file = self.log_dir / 'training_summary.json'
        if is_main_process():
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Training summary saved to {summary_file}")

    def _extract_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize metrics from various formats."""
        return {
            'loss': float(metrics.get('loss', metrics.get('avg_loss', 0))),
            'avg_reward': float(metrics.get('avg_reward', 0)),
            'reward_std': float(metrics.get('reward_std', 0)),
            'avg_kl': float(metrics.get('avg_kl', 0)),
            'avg_grad_norm': float(metrics.get('avg_grad_norm', 0)),
            'max_grad_norm': float(metrics.get('max_grad_norm', 0)),
            'avg_response_length': float(metrics.get('avg_response_length', 0)),
            'num_samples': int(metrics.get('num_samples', 0)),
        }

    def _write_jsonl(self, entry: Dict[str, Any]) -> None:
        """Append entry to JSONL file. Only main process writes to avoid concurrent appends in multi-GPU."""
        if not is_main_process():
            return
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def setup_logging(log_dir: Union[str, Path], log_level: str = "INFO") -> logging.Logger:
    """Setup logging
    
    Note: This function avoids adding duplicate handlers if called multiple times.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance for the current module
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Only configure if no handlers exist (avoid duplicate handlers)
    if not root_logger.handlers:
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    
    return logging.getLogger(__name__)
